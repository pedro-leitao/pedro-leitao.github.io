# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from github import Github, UnknownObjectException
import pandas as pd
from tqdm.auto import tqdm
import requests_cache

def _get_user_data(user, users_data: dict):
    """Safely retrieves user data and handles exceptions for non-existent users."""
    if user and user.login not in users_data:
        try:
            users_data[user.login] = {
                "id": user.id,
                "login": user.login,
                "name": user.name,
                "company": user.company,
                "location": user.location,
                "followers": user.followers,
                "created_at": user.created_at
            }
        except UnknownObjectException:
            print(f"Could not retrieve full profile for user {user.login}. Storing basic info.")
            # Store basic info if the full profile is not available
            users_data[user.login] = {
                "id": user.id,
                "login": user.login,
                "name": None, "company": None, "location": None,
                "followers": -1, "created_at": None
            }

def download_issues(token: str, repo_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download issues from a GitHub repository and return them as DataFrames.
    
    Args:
        token (str): GitHub personal access token.
        repo_name (str): Name of the repository in the format 'owner/repo'.
        
    Returns:
        tuple: DataFrames for issues, comments, users, labels, and events.
    """
    os.makedirs('.data', exist_ok=True)
    requests_cache.install_cache('.data/github_cache', backend='sqlite', expire_after=4*3600)

    g = Github(token)
    if not g:
        raise ValueError("Invalid GitHub token or authentication failed.")

    # 2) Get repo and issues
    repo   = g.get_repo(repo_name)
    if not repo:
        raise ValueError(f"Repository '{repo_name}' not found or access denied.")
    issues = repo.get_issues(state="all")  # Paginated iterator
    if not issues:
        raise ValueError(f"No issues found in repository '{repo_name}'.")

    issue_data = []
    issue_comments = []
    issue_events = []
    users_data = {}
    labels_data = {}

    for issue in tqdm(issues, total=issues.totalCount, desc=f"Downloading issues from {repo_name}"):
        # Add all issue data to list
        issue_data.append(
            {
                "id": issue.id,
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
                "closed_at": issue.closed_at,
                "body": issue.body,
                "labels": [label.name for label in issue.labels],
                "assignees": [assignee.login for assignee in issue.assignees],
                "user": issue.user.login
            }
        )

        # Add user data
        _get_user_data(issue.user, users_data)
        for assignee in issue.assignees:
            _get_user_data(assignee, users_data)

        # Add all comments to list
        for comment in issue.get_comments():
            issue_comments.append(
                {
                    "issue_id": issue.id,
                    "comment_id": comment.id,
                    "user": comment.user.login,
                    "created_at": comment.created_at,
                    "updated_at": comment.updated_at,
                    "body": comment.body
                }
            )
            # Add comment user to users list
            _get_user_data(comment.user, users_data)

        # Add all labels to list
        for label in issue.labels:
            if label.name not in labels_data:
                labels_data[label.name] = {
                    "name": label.name,
                    "color": label.color,
                    "description": label.description
                }

        # Add all events to list
        for event in issue.get_events():
            issue_events.append(
                {
                    "issue_id": issue.id,
                    "event_id": event.id,
                    "actor": event.actor.login if event.actor else None,
                    "event": event.event,
                    "created_at": event.created_at
                }
            )
            # Add event actor to users list
            if event.actor:
                _get_user_data(event.actor, users_data)

    return (
            pd.DataFrame(issue_data),
            pd.DataFrame(issue_comments),
            pd.DataFrame(list(users_data.values())),
            pd.DataFrame(list(labels_data.values())),
            pd.DataFrame(issue_events)
        )
#
#
#
#
#
import os

token = os.getenv("GITHUB_TOKEN")
repo_name = "Farama-Foundation/Gymnasium"

# Check if we already have the data
if os.path.exists(".data/issues.pkl"):
    print("Data already downloaded. Loading from pickle files.")
    issue_data = pd.read_pickle(".data/issues.pkl")
    issue_comments = pd.read_pickle(".data/comments.pkl")
    users_data = pd.read_pickle(".data/users.pkl")
    labels_data = pd.read_pickle(".data/labels.pkl")
    issue_events = pd.read_pickle(".data/events.pkl")
else:
    print("Downloading issues from GitHub...")
    issue_data, issue_comments, users_data, labels_data, issue_events = download_issues(token, repo_name)
    # Save all dataframes to pickle files under `.data`
    os.makedirs(".data", exist_ok=True)
    issue_data.to_pickle(".data/issues.pkl")
    issue_comments.to_pickle(".data/comments.pkl")
    users_data.to_pickle(".data/users.pkl")
    labels_data.to_pickle(".data/labels.pkl")
    issue_events.to_pickle(".data/events.pkl")
#
#
#
#
#
#
#
#
#
#
#
#
#
# Method to compute embeddings using a given Transformer model
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self, model_name: str = "QWen/Qwen3-Embedding-0.6B", batch_size: int = 32, truncate_dim: int = None) -> None:
        # Use CUDA if available
        if torch.cuda.is_available():
            print("Using CUDA for embeddings.")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using MPS for embeddings.")
            self.device = torch.device("mps")
        else:
            print("Using CPU for embeddings.")
            self.device = torch.device("cpu")
        self.model = SentenceTransformer(model_name, truncate_dim=truncate_dim).to(self.device)
        self.batch_size = batch_size

    def embed_batch(self, texts: List[str], desc: str = "Embedding batch") -> np.ndarray:
        """
        Embed a batch of texts using the SentenceTransformer model.

        Args:
            texts (List[str]): List of texts to embed.
            desc (str): Description for the tqdm progress bar.

        Returns:
            np.ndarray: Array of embeddings for the input texts.
        """
        all_embs = []
        self.model.to(self.device)
        with torch.no_grad():  # disable grads
            for i in tqdm(range(0, len(texts), self.batch_size), desc=desc):
                batch = texts[i : i + self.batch_size]
                # get a CPU numpy array directly
                embs = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_tensor=False  # returns numpy on CPU
                )
                all_embs.append(np.vstack(embs) if isinstance(embs, list) else embs)
                # free any CUDA scratch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return np.vstack(all_embs)

embedding_dim = 384  # Set the embedding dimension

embedding_model = EmbeddingModel(batch_size=2, truncate_dim=embedding_dim)
#
#
#
# Compute embeddings for issues and comments, including title and body
def compute_embeddings(df: pd.DataFrame, text_columns: List[str], desc: str = "Computing embeddings") -> np.ndarray:
    """
    Compute embeddings for specified text columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_columns (List[str]): List of column names to compute embeddings for.
        desc (str): Description for the tqdm progress bar.

    Returns:
        np.ndarray: Array of embeddings for the specified text columns.
    """
    texts = []
    for _, row in df.iterrows():
        text = " ".join(str(row[col]) for col in text_columns if pd.notna(row[col]))
        texts.append(text)
    return embedding_model.embed_batch(texts, desc=desc)
#
#
#
#
#
# Check if embeddings already exist
if "embeddings" in issue_data.columns and "embeddings" in issue_comments.columns:
    print("Embeddings already computed. Loading from DataFrame.")
else:
    print("Computing embeddings for issues and comments...")
    issue_text_columns = ["title", "body"]
    issue_embeddings = compute_embeddings(issue_data, issue_text_columns, "Computing issue embeddings")
    comment_text_columns = ["body"]
    comment_embeddings = compute_embeddings(issue_comments, comment_text_columns, "Computing comment embeddings")
    # Add embeddings to DataFrames
    issue_data["embeddings"] = list(issue_embeddings)
    issue_comments["embeddings"] = list(comment_embeddings)
    # Save dataframe back to pickle files
    issue_data.to_pickle(".data/issues.pkl")
    issue_comments.to_pickle(".data/comments.pkl")
#
#
#
#
#
#
#
#
#
# Show a sample for each dataframe
print("Sample issue data:")
issue_data.sample(5)
#
#
#
#
#
print("\nSample issue comments:")
issue_comments.sample(5)
#
#
#
#
#
print("\nSample users data:")
users_data.sample(5)
#
#
#
#
#
print("\nSample labels data:")
labels_data.sample(5)
#
#
#
#
#
print("\nSample issue events:")
issue_events.sample(5)
#
#
#
#
#
#
#
#
#
from neo4j import GraphDatabase, basic_auth, Driver, Session, Transaction, Record
from neo4j.graph import Graph

URI      = os.getenv("NEO4J_URI")
USER     = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
AUTH = (USER, PASSWORD)

print(f"Connecting to Neo4j at {URI} with user {USER}")

driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

def test_aura_connection() -> None:
    with driver.session() as session:
        result = session.run("RETURN 'Hello, Aura!' AS message")
        record = result.single()
        print(record["message"])  # should print "Hello, Aura!"

test_aura_connection()
#
#
#
#
#
def drop_schema(tx: Transaction) -> None:
    # Drop constraints
    for record in tx.run("SHOW CONSTRAINTS"):
        name = record["name"]
        tx.run(f"DROP CONSTRAINT `{name}`")
    # Drop indexes
    for record in tx.run("SHOW INDEXES"):
        name = record["name"]
        tx.run(f"DROP INDEX `{name}`")

def clear_database(tx: Transaction) -> None:
    # Drop all nodes and relationships
    tx.run("MATCH (n) DETACH DELETE n")

def create_constraints(tx: Transaction) -> None:
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label) REQUIRE l.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE")
#
#
#
#
#
def create_issue_vector_index(tx: Transaction, embedding_dim: int = 384) -> None:
    tx.run("""
        CREATE VECTOR INDEX `issue_embeddings` IF NOT EXISTS
        FOR (i:Issue)
        ON i.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: $embedding_dim,
            `vector.similarity_function`: 'cosine'
        }}
    """, embedding_dim=embedding_dim)

def create_comment_vector_index(tx: Transaction, embedding_dim: int = 384) -> None:
    tx.run("""
        CREATE VECTOR INDEX `comment_embeddings` IF NOT EXISTS
        FOR (c:Comment)
        ON c.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: $embedding_dim,
            `vector.similarity_function`: 'cosine'
        }}
    """, embedding_dim=embedding_dim)
#
#
#
with driver.session(database="neo4j") as session:
    # Clear the database
    print("Clearing the database...")
    session.execute_write(clear_database)

    # Drop existing schema
    print("Dropping existing schema...")
    session.execute_write(drop_schema)
    
    # Create new constraints
    print("Creating new constraints...")
    session.execute_write(create_constraints)

    # Create vector indexes
    print("Creating vector indexes...")
    session.execute_write(create_issue_vector_index, embedding_dim=embedding_dim)
    session.execute_write(create_comment_vector_index, embedding_dim=embedding_dim)

    print("Schema updated successfully.")
#
#
#
from typing import List, Dict, Any
import pandas as pd

def _load_users_batch(tx: Transaction, batch: List[Dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $batch AS row
        MERGE (u:User {id: row.id})
        SET u.login = row.login,
            u.name = row.name,
            u.company = row.company,
            u.location = row.location,
            u.followers = row.followers,
            u.created_at = CASE WHEN row.created_at IS NOT NULL THEN datetime(row.created_at) ELSE null END
        """,
        batch=batch
    )

def import_users_batched(session: Session, users_df: pd.DataFrame, batch_size: int = 128) -> None:
    for i in tqdm(range(0, len(users_df), batch_size), desc="Importing users"):
        batch = users_df.iloc[i:i+batch_size].to_dict('records')
        session.execute_write(_load_users_batch, batch)

def _load_labels_batch(tx: Transaction, batch: List[Dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $batch AS row
        MERGE (l:Label {name: row.name})
        SET l.color = row.color,
            l.description = row.description
        """,
        batch=batch
    )

def import_labels_batched(session: Session, labels_df: pd.DataFrame, batch_size: int = 128) -> None:
    for i in tqdm(range(0, len(labels_df), batch_size), desc="Importing labels"):
        batch = labels_df.iloc[i:i+batch_size].to_dict('records')
        session.execute_write(_load_labels_batch, batch)

def _load_issues_batch(tx: Transaction, batch: List[Dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $batch AS row
        MERGE (i:Issue {id: row.id})
        SET i.number = row.number,
            i.title = row.title,
            i.state = row.state,
            i.body = row.body,
            i.created_at = datetime(row.created_at),
            i.updated_at = datetime(row.updated_at),
            i.closed_at = CASE WHEN row.closed_at IS NOT NULL THEN datetime(row.closed_at) ELSE null END,
            i.embedding = row.embeddings
        WITH i, row
        MERGE (u:User {login: row.user})
        MERGE (i)-[:RAISED_BY]->(u)
        WITH i, row
        UNWIND row.labels AS labelName
          MERGE (l:Label {name: labelName})
          MERGE (i)-[:HAS_LABEL]->(l)
        WITH i, row
        UNWIND row.assignees AS assigneeLogin
          MERGE (a:User {login: assigneeLogin})
          MERGE (i)-[:ASSIGNED_TO]->(a)
        """,
        batch=batch
    )

def import_issues_batched(session: Session, issues_df: pd.DataFrame, batch_size: int = 128) -> None:
    for i in tqdm(range(0, len(issues_df), batch_size), desc="Importing issues"):
        batch = issues_df.iloc[i:i+batch_size].to_dict('records')
        session.execute_write(_load_issues_batch, batch)

def _load_comments_batch(tx: Transaction, batch: List[Dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $batch AS row
        MERGE (c:Comment {id: row.comment_id})
        SET c.body = row.body,
            c.created_at = datetime(row.created_at),
            c.updated_at = datetime(row.updated_at),
            c.embedding = row.embeddings
        WITH c, row
        MERGE (i:Issue {id: row.issue_id})
        MERGE (c)-[:COMMENT_ON]->(i)
        WITH c, row
        MERGE (u:User {login: row.user})
        MERGE (c)-[:COMMENT_BY]->(u)
        """,
        batch=batch
    )

def import_comments_batched(session: Session, comments_df: pd.DataFrame, batch_size: int = 128) -> None:
    for i in tqdm(range(0, len(comments_df), batch_size), desc="Importing comments"):
        batch = comments_df.iloc[i:i+batch_size].to_dict('records')
        session.execute_write(_load_comments_batch, batch)

def _load_events_batch(tx: Transaction, batch: List[Dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $batch AS row
        MERGE (e:Event {id: row.event_id})
        SET e.event = row.event,
            e.created_at = datetime(row.created_at)
        WITH e, row
        MERGE (i:Issue {id: row.issue_id})
        MERGE (e)-[:EVENT_ON]->(i)
        WITH e, row
        WHERE row.actor IS NOT NULL
        MERGE (u:User {login: row.actor})
        MERGE (e)-[:EVENT_BY]->(u)
        """,
        batch=batch
    )

def import_events_batched(session: Session, events_df: pd.DataFrame, batch_size: int = 128) -> None:
    for i in tqdm(range(0, len(events_df), batch_size), desc="Importing events"):
        batch = events_df.iloc[i:i+batch_size].to_dict('records')
        session.execute_write(_load_events_batch, batch)
#
#
#
with driver.session() as session:
    # Import data
    print("Importing data...")
    import_users_batched(session, users_data)
    import_labels_batched(session, labels_data)
    import_issues_batched(session, issue_data)
    import_comments_batched(session, issue_comments)
    import_events_batched(session, issue_events)
    print("Data imported successfully.")
#
#
#
from pyvis.network import Network
import pandas as pd
from neo4j.graph import Graph, Node, Relationship

def create_pyvis_network_from_neo4j(graph: Graph) -> Network:
    """
    Creates a Pyvis Network object from a Neo4j graph object.
    """
    net = Network(
        notebook=True,
        cdn_resources='in_line',
        height='750px',
        width='100%',
        bgcolor="#ffffff",
        font_color="black"
    )

    for node in graph.nodes:
        node_id = node.element_id
        labels = list(node.labels)
        group = labels[0] if labels else "Node"
        properties = dict(node)
        
        # Plain‑text title with newlines
        title_lines = [group]
        for k, v in properties.items():
            if k == 'embedding':
                continue
            if k == 'body' and v and len(v) > 512:
                v = v[:512] + "..."
            title_lines.append(f"{k}: {v}")
        title = "\n".join(title_lines)

        # Use a specific property for the label if available
        node_label = str(
            properties.get('title') or
            properties.get('name') or
            properties.get('login') or
            properties.get('id') or
            node_id
        )
        if len(node_label) > 30:
            node_label = node_label[:27] + "..."

        node_size = 25
        if "Issue" in labels:
            # Make the node size relative to the number of related nodes
            related_nodes = len([rel for rel in graph.relationships if rel.start_node.element_id == node_id or rel.end_node.element_id == node_id])
            node_size += related_nodes

        net.add_node(node_id, label=node_label, title=title, group=group, size=node_size)

    # Add edges
    for rel in graph.relationships:
        source_id = rel.start_node.element_id
        target_id = rel.end_node.element_id
        net.add_edge(source_id, target_id, title=rel.type, arrows='to', dashes=True)

    return net
#
#
#
# Query Neo4j to get a sample of the graph data
with driver.session() as session:
    result = session.run("""
    MATCH (i:Issue)
    WITH i, rand() AS r ORDER BY r LIMIT 50
    MATCH (i)-[rel]-(neighbor)
    RETURN i, rel, neighbor
    """)
    graph = result.graph()
    net = create_pyvis_network_from_neo4j(graph)

# Configure physics and controls
net.toggle_physics(True)

# Save the visualization to HTML
net.show("graph_visualization.html", notebook=True)
#
#
#
def create_similarity_links(tx: Transaction, min_score: float) -> int:
    result = tx.run("""
        MATCH (i:Issue)
        CALL db.index.vector.queryNodes('issue_embeddings', 10, i.embedding) YIELD node AS similar_issue, score
        WHERE score >= $min_score AND elementId(i) < elementId(similar_issue)
        MERGE (i)-[r:MIGHT_RELATE_TO]->(similar_issue)
        SET r.score = score
    """, min_score=min_score)
    return result.consume().counters.relationships_created

min_score_threshold = 0.75
with driver.session() as session:
    print(f"Creating MIGHT_RELATE_TO relationships between issues with score >= {min_score_threshold}...")
    num_rels_created = session.execute_write(create_similarity_links, min_score=min_score_threshold)
    print(f"Created {num_rels_created} MIGHT_RELATE_TO relationships.")
#
#
#
import networkx as nx
from networkx.algorithms import community

def create_pyvis_network_from_networkx(G: nx.Graph, node_community: dict, min_score_threshold: float) -> Network:
    """
    Creates a Pyvis Network object from a NetworkX graph object, with community information.
    """
    net = Network(
        notebook=True,
        cdn_resources='in_line',
        height='750px',
        width='100%',
        bgcolor="#ffffff",
        font_color="black"
    )

    # Add nodes to PyVis network with community information
    for node_id, properties in G.nodes(data=True):
        group = node_community.get(node_id, -1)  # -1 for nodes not in any community
        
        # Plain‑text title with newlines
        title_lines = [f"Community: {group}"]
        for k, v in properties.items():
            if k == 'embedding':
                continue
            if k == 'body' and v and len(v) > 512:
                v = v[:512] + "..."
            title_lines.append(f"{k}: {v}")
        title = "\n".join(title_lines)

        # Use a specific property for the label if available
        node_label = str(
            properties.get('title') or
            properties.get('name') or
            properties.get('login') or
            properties.get('id') or
            node_id
        )
        if len(node_label) > 30:
            node_label = node_label[:27] + "..."

        net.add_node(node_id, label=node_label, title=title, group=group)

    # Add edges
    for source_id, target_id, properties in G.edges(data=True):
        rel_title = properties.get('type', '')
        edge_width = 1
        if 'score' in properties:
            score = properties['score']
            rel_title = f"MIGHT_RELATE_TO (score: {score:.2f})"
            # Scale edge width based on score.
            edge_width = 1 + (score - min_score_threshold) * (10 / (1 - min_score_threshold))
            
        net.add_edge(source_id, target_id, title=rel_title, width=edge_width, arrows='to', dashes=True)

    return net
#
#
#
# Create a NetworkX graph to perform community detection
G = nx.Graph()

# Query Neo4j to get a sample of issues with MIGHT_RELATE_TO relationships
with driver.session() as session:
    result = session.run("""
    MATCH (i:Issue)-[rel:MIGHT_RELATE_TO]-(neighbor:Issue)
    WITH i, rel, neighbor, rand() as r
    ORDER BY r
    LIMIT 200
    RETURN i, rel, neighbor
    """)
    
    # Build the NetworkX graph from the query results
    for record in result:
        node_i = record["i"]
        node_neighbor = record["neighbor"]
        rel = record["rel"]
        
        G.add_node(node_i.element_id, **dict(node_i))
        G.add_node(node_neighbor.element_id, **dict(node_neighbor))
        G.add_edge(node_i.element_id, node_neighbor.element_id, **dict(rel))

# Detect communities using the Louvain method
communities = community.louvain_communities(G)
# Create a mapping from node to community id
node_community = {}
for i, comm in enumerate(communities):
    for node_id in comm:
        node_community[node_id] = i

net_similar = create_pyvis_network_from_networkx(G, node_community, min_score_threshold)

# Configure physics and controls
net_similar.toggle_physics(True)

# Save the visualization to HTML
net_similar.show("might_relate_to_visualization.html", notebook=True)
#
#
#
def get_rag_graph(tx: Transaction, query_string: str, top_k: int = 5) -> Graph:
    """
    Finds the most relevant issues to a query string by searching both issue and comment embeddings,
    and returns a graph of their connections.

    The graph contains:
    - The top-k matching issues based on a blended search of issues and comments.
    - For each of these issues: their comments, users who wrote them, and labels.
    """
    # Embed the query string
    query_embedding = embedding_model.embed_batch([query_string])[0].tolist()

    # Find the most relevant issues and build the graph
    result = tx.run("""
        // Find top k issues from issue embeddings and from comment embeddings
        CALL {
            CALL db.index.vector.queryNodes('issue_embeddings', $top_k, $embedding) YIELD node AS issue, score
            RETURN issue, score
            UNION
            CALL db.index.vector.queryNodes('comment_embeddings', $top_k, $embedding) YIELD node AS comment, score
            MATCH (comment)-[:COMMENT_ON]->(issue:Issue)
            RETURN issue, score
        }
        
        // Combine, deduplicate, and select top k issues overall
        WITH issue, score
        ORDER BY score DESC
        WITH collect(issue {.*, score: score}) AS issues
        WITH [i in issues | i.id] AS issueIds, issues
        WITH [id IN issueIds | head([i IN issues WHERE i.id = id])] AS uniqueIssues
        WITH uniqueIssues[..$top_k] AS top_issues
        UNWIND top_issues as top_issue_data
        MATCH (top_issue:Issue {id: top_issue_data.id})

        // Collect the top issues, their labels, and the users who raised them
        OPTIONAL MATCH (top_issue)-[r1:HAS_LABEL|RAISED_BY]->(n1)

        // Collect comments on the top issues and the users who made them
        OPTIONAL MATCH (top_issue)<-[r2:COMMENT_ON]-(c1:Comment)-[r3:COMMENT_BY]->(u1:User)
        
        // Aggregate all nodes and relationships per issue
        WITH top_issue, 
             collect(DISTINCT n1) as nodes1,
             collect(DISTINCT r1) as rels1,
             collect(DISTINCT c1) + collect(DISTINCT u1) as nodes2,
             collect(DISTINCT r2) + collect(DISTINCT r3) as rels2

        // Aggregate all nodes and relationships across all issues
        WITH collect(top_issue) + apoc.coll.flatten(collect(nodes1)) + apoc.coll.flatten(collect(nodes2)) as all_nodes,
             apoc.coll.flatten(collect(rels1)) + apoc.coll.flatten(collect(rels2)) as all_rels

        UNWIND all_nodes as n
        UNWIND all_rels as r
        RETURN collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships
    """, embedding=query_embedding, top_k=top_k)
    
    record = result.single()
    
    # Reconstruct the graph from nodes and relationships
    nodes = record["nodes"]
    relationships = record["relationships"]
    
    # Create a graph object to return
    # This is a bit of a hack, as we can't directly instantiate a Graph object easily
    # with nodes and relationships from the driver. We'll run a query that returns a graph.
    if not nodes:
        return Graph()

    node_ids = [n.element_id for n in nodes]
    
    graph_result = tx.run("""
        MATCH (n) WHERE elementId(n) IN $node_ids
        OPTIONAL MATCH (n)-[r]-(m) WHERE elementId(n) IN $node_ids AND elementId(m) IN $node_ids
        RETURN n, r, m
    """, node_ids=node_ids)
    
    return graph_result.graph()
#
#
#
query_string = "What are the dependencies necessary to run Atari environments ?"
with driver.session() as session:
    print(f"Finding RAG graph for query: {query_string}")
    rag_graph = session.execute_read(get_rag_graph, query_string)
    print(f"Found {len(rag_graph.nodes)} nodes and {len(rag_graph.relationships)} relationships in the RAG graph.")
#
#
#
# Visualize the RAG graph using Pyvis
rag_net = create_pyvis_network_from_neo4j(rag_graph)
rag_net.toggle_physics(True)
rag_net.show("rag_graph_visualization.html", notebook=True)
#
#
#
from google import genai

# Configure the Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai_client = genai.Client(api_key=gemini_api_key)
#
#
#
def graph_to_textual_summary(graph: Graph) -> str:
    """Converts a Neo4j graph to a textual summary for an LLM."""
    if not graph.nodes:
        return "No information found for the query."

    summary = "Found the following information:\n\n"
    
    nodes_summary = "Nodes:\n"
    for node in graph.nodes:
        labels = list(node.labels)
        properties = dict(node)
        
        # Create a representative name for the node
        node_name_parts = []
        if 'title' in properties: node_name_parts.append(f"title='{properties['title']}'")
        if 'name' in properties: node_name_parts.append(f"name='{properties['name']}'")
        if 'login' in properties: node_name_parts.append(f"login='{properties['login']}'")
        if 'id' in properties: node_name_parts.append(f"id={properties['id']}")
        node_name = f"({', '.join(node_name_parts)})"

        nodes_summary += f"- Node {node_name} (Labels: {labels}):\n"
        for k, v in properties.items():
            if k in ['embedding', 'title', 'name', 'login', 'id']:
                continue
            if k == 'body' and v and len(v) > 200:
                v = v[:200] + "..."
            nodes_summary += f"  - {k}: {v}\n"
    
    rels_summary = "\nRelationships:\n"
    for rel in graph.relationships:
        start_node_props = dict(rel.start_node)
        end_node_props = dict(rel.end_node)
        
        start_node_id = start_node_props.get('title') or start_node_props.get('name') or start_node_props.get('login') or start_node_props.get('id')
        end_node_id = end_node_props.get('title') or end_node_props.get('name') or end_node_props.get('login') or end_node_props.get('id')
        
        rels_summary += f"- ({start_node_id}) -[{rel.type}]-> ({end_node_id})\n"

    return summary + nodes_summary + rels_summary
#
#
#

def find_issues_from_prompt(query_string: str) -> dict:
    """
    Finds potential issues from a user prompt, gets the graph matching the prompt,
    and returns a textual summary of the graph.
    
    Args:
        query_string: The user's query about issues.
        
    Returns:
        A dictionary containing a summary of the retrieved graph data.
    """
    print(f"Agent is calling find_issues_from_prompt with query: '{query_string}'")
    with driver.session() as session:
        rag_graph = session.execute_read(get_rag_graph, query_string)
        if rag_graph:
            print(f"Found {len(rag_graph.nodes)} nodes and {len(rag_graph.relationships)} relationships.")
            summary = graph_to_textual_summary(rag_graph)
            return {"summary": summary}
        else:
            return {"summary": "Could not find any relevant information in the graph."}
#
#
#
def find_experts(query_string: str) -> dict:
    """
    Finds potential experts on a topic by analyzing who has contributed to the most relevant issues.
    
    Args:
        query_string: The user's query describing the topic of interest.
        
    Returns:
        A dictionary containing a summary of potential experts.
    """
    print(f"Agent is calling find_experts with query: '{query_string}'")
    
    # 1. Embed the query string
    query_embedding = embedding_model.embed_batch([query_string])[0].tolist()

    # 2. Find experts in the graph
    with driver.session() as session:
        result = session.run("""
            // Find the top matching issue for the query embedding
            CALL db.index.vector.queryNodes('issue_embeddings', 1, $embedding) YIELD node AS top_issue
            
            // Collect the top issue and up to 5 of its most similar issues
            WITH top_issue
            OPTIONAL MATCH (top_issue)-[r:MIGHT_RELATE_TO]-(related_issue:Issue)
            WITH top_issue, related_issue, r.score as score
            ORDER BY score DESC
            WITH top_issue, collect(related_issue)[..5] AS related_issues
            WITH [top_issue] + related_issues AS all_issues
            UNWIND all_issues as issue

            // Find all users who have interacted with these issues
            OPTIONAL MATCH (issue)<-[:RAISED_BY]-(u1:User)
            OPTIONAL MATCH (issue)<-[:ASSIGNED_TO]-(u2:User)
            OPTIONAL MATCH (issue)<-[:COMMENT_ON]-(:Comment)-[:COMMENT_BY]->(u3:User)

            // Aggregate and rank the users
            WITH issue, u1, u2, u3
            WITH collect(u1) + collect(u2) + collect(u3) as users, issue
            UNWIND users as user
            WITH user, count(issue) as interactions, collect(DISTINCT {id: issue.id, title: issue.title}) as issues
            ORDER BY interactions DESC
            LIMIT 5
            
            RETURN collect({user: user.login, interactions: interactions, issues: issues}) as experts
        """, embedding=query_embedding)
        
        experts = result.single()["experts"]
        
        if experts:
            summary = "Found the following potential experts based on their interactions with relevant issues:\n\n"
            for expert in experts:
                summary += f"- User: {expert['user']} (Interactions: {expert['interactions']})\n"
                for issue in expert['issues']:
                    summary += f"  - Interacted with issue #{issue['id']}: {issue['title']}\n"
            return {"summary": summary}
        else:
            return {"summary": "Could not find any potential experts for this topic."}
#
#
#
from google.genai import types
from IPython.display import display, Markdown

def converse_with_agent(user_prompt: str) -> str:
    """
    Converse with the agent using a user prompt.
    
    Args:
        user_prompt: The user's query to the agent.
        
    Returns:
        The agent's response as a string.
    """
    config = types.GenerateContentConfig(
        system_instruction=f'''You are an expert agent that can find issues in a Neo4j graph database based on user prompts for the {repo_name} repository. Use the find_issues_from_prompt tool to retrieve relevant issues, summarize them into two sections:
        
        ### Summary
        Provide a concise summary of the issues found in the graph based on the user prompt.
        ### Potential Issues
        List the potential issues that match the user prompt, including relevant details such as issue titles, labels, and any other pertinent information.
        ### Advice
        Provide any advice or recommendations based on the issues found in the graph.

        When using the find_experts tool, summarize the potential experts based on their interactions with relevant issues as a table, including their usernames and interaction counts, and a list of relevant issue titles and ID's.

        Strictly use only the information provided by the tool to formulate your response.''',
        temperature=0.4,
        tools=[find_issues_from_prompt, find_experts]
    )
    
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        config=config,
        contents=user_prompt
    )

    if not response.candidates or not response.candidates[0].content.parts:
        return "No response generated by the agent."
    
    return response.candidates[0].content.parts[0].text

#
#
#
user_prompt = "What are the dependencies necessary to run Atari environments ?"
print(f"User prompt: {user_prompt}\n")

response = converse_with_agent(user_prompt)

print("\nAgent Response:")

boxed_md = f"""
::: callout-note
{response}
:::
"""

display(Markdown(boxed_md))
#
#
#
user_prompt = "How do I make sure I completely seed the environment before running an experiment ?"

print(f"User prompt: {user_prompt}\n")

response = converse_with_agent(user_prompt)

print("\nAgent Response:")

boxed_md = f"""
::: callout-note
{response}
:::
"""

display(Markdown(boxed_md))
#
#
#
user_prompt = "Who could I reach out to for help with an issue with CarRacing environments ?"

print(f"User prompt: {user_prompt}\n")

response = converse_with_agent(user_prompt)

print("\nAgent Response:")

boxed_md = f"""
::: callout-note
{response}
:::
"""

display(Markdown(boxed_md))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
