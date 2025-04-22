# The Reward Alchemist: Hackathon Write-Up (Rayv Code-o-Tron 3000)

**Team:** Phantom Troupe (Ishwar, Vamshi, Nachiketh)
**Problem Statement:** #2 - Recommendation Engine to Target Rewards Based on User Persona

**Introduction & Problem:**

The digital landscape bombards users with generic offers, leading to low engagement and wasted resources for businesses. Traditional recommendation systems often fail to capture the nuanced personas, local context, and specific interests prevalent in diverse markets like India. Addressing Problem Statement #2 for the Rayv Code-o-Tron 3000, our project, **The Reward Alchemist**, tackles this challenge by creating a hyper-personalized campaign recommendation engine designed to understand users deeply and suggest rewards they will truly value, aiming to exceed the goal of 80% simulated user satisfaction.

**Our Approach: Hybrid AI for Contextual Relevance**

We developed a sophisticated **hybrid recommendation strategy** combining multiple AI techniques:

1.  **Graph Neural Networks (GNNs - GraphSAGE):** We modeled the user-campaign ecosystem as a graph, allowing GraphSAGE to learn rich embeddings that capture both explicit features (user demographics, campaign details) and implicit relational patterns (collaborative filtering effects). This provides a powerful foundation for understanding overall user affinity.
2.  **Semantic Understanding (SBERT):** To capture the meaning behind user interests/watch history and campaign text (promos/categories), we integrated Sentence-BERT embeddings. Cosine similarity between user text embeddings and campaign text embeddings allows for nuanced matching beyond simple keywords.
3.  **Multi-Stage Ranking Logic:** Recognizing that raw model scores need context, we implemented a prioritized ranking algorithm in the final application:
    *   **Location First:** Campaigns matching the user's exact location are ranked highest, followed by those in the same geographic region.
    *   **Semantic Sorting:** Within these location tiers, results are further sorted based on SBERT similarity scores, prioritizing recent watch history alignment over general interest alignment.
    *   This ensures recommendations are geographically relevant *and* personally meaningful.

**Key Innovations & Features:**

*   **Hybrid Architecture:** The synergistic use of GNNs and SBERT provides a more robust understanding than either method alone.
*   **Explainable AI:** Each recommendation includes a "Reason" derived from the ranking logic (e.g., "Exact Location Match + Strong Watch History Match"), enhancing transparency.
*   **Dynamic User Tiering:** We implemented the stretch goal, classifying users (Basic/Premium/Elite) based on their activity level and average GNN relevance score, enabling potential future targeted reward strategies.
*   **Realistic Data Context:** Trained on a detailed synthetic dataset simulating 'Desi' user personas.
*   **On-Demand Processing:** Embeddings are computed on-the-fly in the Flask app for demo stability.
*   **Targeted Satisfaction Metric:** Our application calculates a weighted User Satisfaction Score based on the final ranked list's relevance and proximity, directly addressing the hackathon's core goal.
*   **Minimalist Dark UI:** A clean, modern Flask web interface.

**Training Evaluation Results:**

The underlying GNN model demonstrated strong performance in learning user-campaign affinities on the held-out test set generated during training:
*   **Test AUC: 0.9896** (Excellent ability to distinguish relevant from non-relevant pairs).
*   *Baseline GNN Top-10 Metrics:* Test Precision@10: 0.0739, Test NDCG@10: 0.0993.
*(Note: P@10/NDCG@10 measure raw GNN link prediction accuracy *before* applying the crucial location and semantic ranking logic used in the final application, which prioritizes practical user satisfaction).*

**Challenges & Learnings:**

*   **Data Handling:** Preprocessing and feature engineering for the diverse dataset.
*   **Ranking Nuance:** Designing the multi-stage hybrid ranking to balance GNN scores, location, and semantic signals effectively was key. While baseline GNN metrics (P@10/NDCG@10) were modest, the high AUC indicated strong potential, which the application-level ranking aims to translate into user satisfaction.
*   **Performance Trade-offs:** Opting for on-demand embeddings for demo stability vs. pre-computation for production speed.

**Why It's Awesome:**

The Reward Alchemist moves beyond basic recommendations. By intelligently combining graph-based learning (strong AUC), semantic understanding, and essential contextual rules (location), it delivers recommendations that feel intuitive and genuinely personalized. The explainability fosters trust, the dynamic tiering adds user insight, and the focus on a practical satisfaction metric directly addresses the hackathon challenge with an effective, well-executed solution.

---

**Acknowledgments:**

Portions of the code structure, explanations, frontend refinements, and documentation were developed with the assistance of AI models, primarily Google's Gemini and Anthropic's Claude, used iteratively for ideation, debugging, and code generation support throughout the hackathon process.

