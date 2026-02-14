import faiss
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
import anthropic
import requests
import json
from datetime import datetime
import re

# Load index + metadata
@st.cache_resource
def load_data():
    index = faiss.read_index("factcheck_index.faiss")
    metadata = pd.read_json("factcheck_metadata.json")
    return index, metadata

index, metadata = load_data()

# API clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
claude_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Google Fact Check API Key
GOOGLE_FACTCHECK_API_KEY = st.secrets["GOOGLE_FACTCHECK_API_KEY"]

# Embed user query
@st.cache_data
def embed_text(text):
    response = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Multi-source fact checking
def search_google_factcheck(query, max_results=5):
    """Search Google's Fact Check Tools API"""
    if not GOOGLE_FACTCHECK_API_KEY:
        return []
    
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            'query': query,
            'key': GOOGLE_FACTCHECK_API_KEY,
            'pageSize': max_results
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            claims = data.get('claims', [])
            
            results = []
            for claim in claims:
                claim_reviews = claim.get('claimReview', [])
                for review in claim_reviews:
                    results.append({
                        'source': 'Multi-Source',
                        'publisher': review.get('publisher', {}).get('name', 'Unknown'),
                        'claim': claim.get('text', ''),
                        'rating': review.get('textualRating', 'No rating'),
                        'url': review.get('url', ''),
                        'review_date': review.get('reviewDate', ''),
                        'language': review.get('languageCode', 'en')
                    })
            return results[:max_results]
        else:
            st.warning(f"Google Fact Check API returned status {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Error accessing Google Fact Check API: {str(e)}")
        return []

def standardize_rating(rating, source='unknown'):
    """Convert various rating systems to a 0-5 scale"""
    rating = str(rating).lower().strip()
    
    # PolitiFact scale
    politifact_map = {
        'true': 5,
        'mostly true': 4,
        'half true': 3,
        'mostly false': 2,
        'false': 1,
        'pants on fire': 0
    }
    
    # Generic mappings for other sources
    generic_map = {
        'true': 5,
        'mostly true': 4,
        'mixture': 3,
        'mixed': 3,
        'partly false': 3,
        'mostly false': 2,
        'false': 1,
        'fake': 1,
        'misleading': 2,
        'unproven': 3,
        'correct': 5,
        'incorrect': 1,
        'accurate': 5,
        'inaccurate': 1
    }
    
    # Try PolitiFact first, then generic
    for rating_text, score in {**politifact_map, **generic_map}.items():
        if rating_text in rating:
            return score
    
    return None  # Unknown rating

def analyze_consensus(sources_data):
    """Analyze agreement between different fact-checking sources"""
    if not sources_data:
        return {
            'consensus_level': 0,
            'average_rating': None,
            'agreement': 'No data',
            'outliers': [],
            'source_count': 0
        }
    
    # Extract standardized ratings
    ratings = []
    source_ratings = {}
    
    for source_data in sources_data:
        for result in source_data.get('results', []):
            rating = standardize_rating(result.get('rating', ''))
            if rating is not None:
                ratings.append(rating)
                source_name = result.get('publisher', result.get('source', 'Unknown'))
                source_ratings[source_name] = rating
    
    if not ratings:
        return {
            'consensus_level': 0,
            'average_rating': None,
            'agreement': 'No standardizable ratings',
            'outliers': [],
            'source_count': len(sources_data)
        }
    
    # Calculate consensus metrics
    avg_rating = np.mean(ratings)
    std_rating = np.std(ratings) if len(ratings) > 1 else 0
    
    # Consensus level (higher when ratings are similar)
    consensus_level = max(0, 1 - (std_rating / 2.5))  # Normalize by max possible std
    
    # Identify outliers (ratings far from average)
    outliers = []
    for source, rating in source_ratings.items():
        if abs(rating - avg_rating) > 1.5:  # More than 1.5 points from average
            outliers.append(f"{source} ({rating})")
    
    # Agreement description
    if consensus_level > 0.8:
        agreement = "Strong consensus"
    elif consensus_level > 0.6:
        agreement = "Moderate agreement"
    elif consensus_level > 0.4:
        agreement = "Some disagreement"
    else:
        agreement = "Significant disagreement"
    
    return {
        'consensus_level': consensus_level,
        'average_rating': avg_rating,
        'agreement': agreement,
        'outliers': outliers,
        'source_count': len(set(source_ratings.keys())),
        'source_ratings': source_ratings
    }

def search_politifact_db(query):
    """Search the local PolitiFact database"""
    query_vector = np.array([embed_text(query)], dtype="float32")
    D, I = index.search(query_vector, k=5)
    top_matches = metadata.iloc[I[0]]
    
    results = []
    for _, row in top_matches.iterrows():
        results.append({
            'source': 'PolitiFact',
            'publisher': 'PolitiFact',
            'claim': row['claim'],
            'rating': row['rating'],
            'explanation': row.get('explanation', '')[:500] + "..." if len(row.get('explanation', '')) > 500 else row.get('explanation', ''),
            'url': row['url'],
            'similarity_score': float(D[0][list(top_matches.index).index(row.name)])
        })
    
    return {'source_name': 'PolitiFact Database', 'results': results}

def get_multi_source_analysis(query):
    """Get fact-checks from multiple sources and analyze consensus"""
    sources_data = []
    
    # Search PolitiFact database
    politifact_data = search_politifact_db(query)
    sources_data.append(politifact_data)
    
    # Search Google Fact Check API
    google_results = search_google_factcheck(query)
    if google_results:
        google_data = {
            'source_name': 'External Sources (via Google)',
            'results': google_results
        }
        sources_data.append(google_data)
    
    # Analyze consensus
    consensus = analyze_consensus(sources_data)
    
    return sources_data, consensus

# Enhanced prompt building
def build_enhanced_prompt(query, sources_data, consensus):
    """Build prompt with multi-source context and consensus analysis"""
    
    # Build context from all sources
    all_contexts = []
    
    for source_data in sources_data:
        source_name = source_data.get('source_name', 'Unknown Source')
        results = source_data.get('results', [])
        
        if results:
            all_contexts.append(f"\n--- {source_name} ---")
            for i, result in enumerate(results[:3]):  # Limit to top 3 per source
                claim = result.get('claim', 'No claim text')
                rating = result.get('rating', 'No rating')
                explanation = result.get('explanation', '')
                url = result.get('url', '')
                publisher = result.get('publisher', source_name)
                
                context_text = f"Claim: {claim}\nPublisher: {publisher}\nRating: {rating}"
                if explanation:
                    context_text += f"\nExplanation: {explanation}"
                if url:
                    context_text += f"\nURL: {url}"
                
                all_contexts.append(context_text)
    
    combined_context = "\n\n".join(all_contexts)
    
    # Add consensus analysis
    consensus_text = f"""
    
CROSS-SOURCE CONSENSUS ANALYSIS:
- Agreement Level: {consensus['agreement']}
- Sources Analyzed: {consensus['source_count']}
- Average Rating Tendency: {consensus.get('average_rating', 'N/A')}
"""
    
    if consensus.get('outliers'):
        consensus_text += f"- Outlying Opinions: {', '.join(consensus['outliers'])}"
    
    return f"""
You are a senior research assistant helping a journalist assess the accuracy of a draft article for PolitiFact.

Your task is to:
1. Analyze the claim(s) in the draft text.
2. Compare the evidence in the article to relevant fact-checks from MULTIPLE sources.
3. Consider both the **evidence in the article** and the **ratings and reasoning** from similar fact-checks.
4. Pay special attention to the **consensus analysis** - note areas of agreement or disagreement between sources.
5. If sources disagree significantly, explain possible reasons and which evidence seems strongest.
6. If the article's claims appear **novel** or significantly **different from precedent**, state that clearly.

Return your assessment in the following format:

**Suggested Rating:**
[Your recommendation, e.g. False, Mostly False, Half True, Mostly True, True, etc.]

**Confidence Level:**
[High/Medium/Low - based on quality and consensus of evidence]

**Reasoning:**
Explain your reasoning using both the factual evidence provided in the draft article and the past fact-checks retrieved from multiple sources. Consider the consensus analysis and any disagreements between sources.

**Multi-Source Analysis:**
Summarize how different fact-checking organizations have approached similar claims. Note areas of consensus and disagreement.

**Jurisprudence:**
List the most similar claims from past fact-checks and how they were rated by different organizations. If none are relevant, say so.

**Evidence Gaps:**
What additional reporting or verification would strengthen this fact-check?

---
Draft article to fact-check:
{query}

---
Relevant fact-checks from multiple sources:
{combined_context}

{consensus_text}
"""

# Streamlit UI
st.title("PolitiFact Rating Recommender")
st.caption("This generativr AI tool compares draft articles against PolitiFact's archive and external fact-checking sources. It identifies relevant jurisprudence, analyzes cross-source consensus and provides structured recommendations to support editorial decisions. I used retrieval-augmented generation (RAG) to mostly eliminate hallucinations by tying it to fact-check databases. Note: This is a prototype based on 9,000 fact-checks, so answers will not be complete.")

# Feature info in sidebar
with st.sidebar:
    st.header("⚙️ Features")
    st.success("✅ Google Fact Check API configured")
    st.markdown("**Capabilities:**")
    st.markdown("• PolitiFact database search")
    st.markdown("• Multi-source fact-check retrieval")
    st.markdown("• Consensus analysis")
    st.markdown("• Confidence scoring")
    st.markdown("• Evidence gap identification")

# Main interface
query = st.text_area(
    "📝 Paste your draft article or claim to fact-check:", 
    placeholder="e.g., 'The president said unemployment is at a historic low of 3.2%'",
    height=150
)

if query:
    with st.spinner("🔍 Analyzing across multiple fact-checking sources..."):
        # Get multi-source analysis
        sources_data, consensus = get_multi_source_analysis(query)
        
        # Generate enhanced prompt and get Claude's analysis
        prompt = build_enhanced_prompt(query, sources_data, consensus)
        
        try:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            
            # Display the main analysis
            st.markdown("## ✨ Analysis ✅")
            st.markdown(response.content[0].text.strip())
            
        except Exception as e:
            st.error(f"Error generating analysis: {str(e)}")
    
    # Display consensus metrics
    if consensus['source_count'] > 1:
        st.markdown("## Cross-Source Consensus")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            consensus_color = "green" if consensus['consensus_level'] > 0.7 else "orange" if consensus['consensus_level'] > 0.4 else "red"
            st.metric("Agreement Level", consensus['agreement'])
        
        with col2:
            st.metric("Sources Found", consensus['source_count'])
        
        with col3:
            if consensus.get('average_rating') is not None:
                rating_labels = {5: "True", 4: "Mostly True", 3: "Mixed", 2: "Mostly False", 1: "False", 0: "Pants on Fire"}
                avg_label = rating_labels.get(round(consensus['average_rating']), f"{consensus['average_rating']:.1f}")
                st.metric("Average Tendency", avg_label)
        
        if consensus.get('outliers'):
            st.warning(f"**Outlying opinions:** {', '.join(consensus['outliers'])}")
    
    # Display source details
    st.markdown("## Retrieved fact-checks by source")
    
    for source_data in sources_data:
        source_name = source_data.get('source_name', 'Unknown Source')
        results = source_data.get('results', [])
        
        if results:
            with st.expander(f"{source_name} ({len(results)} results)"):
                for i, result in enumerate(results):
                    claim = result.get('claim', 'No claim text')
                    rating = result.get('rating', 'No rating')
                    publisher = result.get('publisher', 'Unknown')
                    url = result.get('url', '')
                    
                    # Format the display
                    st.markdown(f"**{i+1}. {claim[:100]}{'...' if len(claim) > 100 else ''}**")
                    st.markdown(f"*Publisher:* {publisher} | *Rating:* {rating}")
                    
                    if url:
                        st.markdown(f"[📖 Read full fact-check]({url})")
                    
                    # Show similarity score for PolitiFact results
                    if 'similarity_score' in result:
                        st.caption(f"Similarity score: {result['similarity_score']:.3f}")
                    
                    if i < len(results) - 1:
                        st.markdown("---")

# Footer
st.markdown("---")
st.markdown("**Important:** This tool provides AI-powered recommendations to support human fact-checkers. Always verify claims with primary sources and apply editorial judgment.")