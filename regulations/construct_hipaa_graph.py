import sys
sys.path.append('..')
from bs4 import BeautifulSoup
import networkx as nx
import json
from utils import *

def match_tag(tag):
    """
    This function is used to filter out the tags that we don't want to keep.
    """
    return (tag.name == 'h1' and tag.has_attr('data-hierarchy-metadata')) \
           or (tag.name == 'h2' and tag.has_attr('data-hierarchy-metadata'))\
           or (tag.name == 'h4' and tag.has_attr('data-hierarchy-metadata'))\
           or (tag.name == 'p' and 'indent-1' in tag.get('class', []))\
           or (tag.name == 'p' and 'indent-2' in tag.get('class', []))\
           or (tag.name == 'p' and 'indent-3' in tag.get('class', []))\
           or (tag.name == 'p' and 'indent-4' in tag.get('class', []))\
           or (tag.name == 'p' and 'indent-5' in tag.get('class', []))\
           or (tag.name == 'p' and 'indent-6' in tag.get('class', []))\
           or (tag.name == 'p' and len(tag.attrs) == 0)

def get_element_from_html(input_file = "legislation/subchapter-C_part-160.html"):
    """
    This function is used to get the elements from the html file.
    """
    with open(input_file, 'r') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    elements = soup.find_all(match_tag)
    for i, element in enumerate(elements):
        if element.name == 'h1':
            index = i
            break
    elements = elements[index:]
    return elements

def generate_graph_from_elements(elements):
    G = nx.DiGraph()
    level = {
        "h1": 0,
        "h2": 1,
        "h4": 2,
        "indent-1": 3,
        "indent-2": 4,
        "indent-3": 5,
        "indent-4": 6,
        "indent-5": 7,
        "indent-6": 8
    }
    level_status = {
        "h1": None,
        "h2": None,
        "h4": None,
        "indent-1": None,
        "indent-2": None,
        "indent-3": None,
        "indent-4": None,
        "indent-5": None,
        "indent-6": None
    }
    nearest_id = None
    for e in elements:
        e_id = None
        e_text = None
        e_name = e.name
        if e.get("data-hierarchy-metadata"):
            metadata = json.loads(e.get("data-hierarchy-metadata"))
            e_id = metadata.get("citation")
        elif e.get("class") and e.get("data-title"):
            e_name = e.get("class")[0]
            e_id = e.get("data-title")
        elif e.get("class") and e.get("id"):
            e_name = e.get("class")[0]
            e_id = e.get("id").replace("p-", "")
        e_text = e.get_text()
        if not e_id:
            if e_name == "p":
                e_text = G.nodes[nearest_id]["text"] + " " + e_text
                G.nodes[nearest_id]["text"] = e_text
                continue
        else:
            e_id = e_id.replace("45 CFR", "").replace(" ","")
            nearest_id = e_id
        e_id = e_id.replace("<em>", "").replace("</em>", "")
        e_id = e_id.replace("“", "").replace("”", "")
        level_status[e_name] = e_id
        curr_level = level[e_name]
        for x, y in level.items():
            if y > curr_level:
                level_status[x] = None

        # e_id = e_id.replace("“", "").replace("”", "")
        
        G.add_node(e_id, text=e_text)
        prev_level = curr_level - 1 if curr_level > 0 else 0

        if curr_level > 0:
            prev_id = level_status[list(level.keys())[list(level.values()).index(prev_level)]]
            if "164.501" in e_id:
                print(prev_id)
                print(e_id)
            G.add_edge(prev_id, e_id, relation = "subsume")
            G.add_edge(e_id, prev_id, relation = "subsumedBy")
    return G

def combine_graphs(Graphs: list,
                   nodes: list,
                   top_id="HIPAA",
                   top_text="HIPAA Privacy Rule"):
    """
    Combine list of graphs.
    """
    G = nx.DiGraph()
    G.add_node(top_id, text=top_text)
    for g in Graphs:
        G.add_nodes_from(g.nodes(data=True))
        G.add_edges_from(g.edges(data=True))
    for node in nodes:
        G.add_edge("HIPAA", node, relation = "subsume")
        G.add_edge(node, "HIPAA", relation = "subsumedBy")
    return G

def generate_graph_from_htmls(input_files, output_graph_file, nodes):
    """
    generate graph of all HIPAA Administrative Simplification Regulations from html files.
    """
    Graphs = []
    for file in input_files:
        elements = get_element_from_html(file)
        Graphs.append(generate_graph_from_elements(elements))
    combined_graph = combine_graphs(Graphs, nodes)
    for file in input_files:
        elements = get_element_from_html(file)
        combined_graph = generate_reference_from_htmls(combined_graph, elements)
    nx.write_graphml(combined_graph, output_graph_file)
    return combined_graph

def filter_164_partAE(elements):
    """
    The Privacy Rule is located at 45 CFR Part 160 and Subparts A and E of Part 164. 
    """
    subpartA_end = 0
    subpartE_start = 0
    for i, element in enumerate(elements):
        if element.name == 'h2':
            if "Subpart B" in element.get_text().strip():
                subpartA_end = i
            if "Subpart E" in element.get_text().strip():
                subpartE_start = i
    elements = elements[0:subpartA_end] + elements[subpartE_start:]
    return elements

def generate_hpr_graph_from_htmls(input_files, output_graph_file, nodes):
    '''
    generate HIPAA privacy rule graph from html files
    '''
    Graphs = []
    for file in input_files:
        elements = get_element_from_html(file)
        if file == 'subchapter-C_part-164.html':
            elements = filter_164_partAE(elements)
        Graphs.append(generate_graph_from_elements(elements))
    combined_graph = combine_graphs(Graphs, nodes)
    for file in input_files:
        elements = get_element_from_html(file)
        if file == 'subchapter-C_part-164.html':
            elements = filter_164_partAE(elements)
        combined_graph = generate_reference_from_htmls(combined_graph, elements)
    nx.write_graphml(combined_graph, output_graph_file)
    return combined_graph

def generate_reference_from_htmls(G, elements):
    e_id = None
    for e in elements:
        e_name = e.name
        if e.get("data-hierarchy-metadata"):
            metadata = json.loads(e.get("data-hierarchy-metadata"))
            e_id = metadata.get("citation")
        elif e.get("class") and e.get("data-title"):
            e_name = e.get("class")[0]
            e_id = e.get("data-title")
        elif e.get("class") and e.get("id"):
            e_name = e.get("class")[0]
            e_id = e.get("id").replace("p-", "")
        elif e_name == "p":
            e_id = e_id
        e_id = e_id.replace("<em>", "").replace("</em>", "")
        if e.find_all('a', class_='cfr'):
            hrefs = e.find_all('a', class_='cfr')
            for href in hrefs:
                href = href.get('href')
                if "#" in href:
                    href = href.split("#")[1]
                    href = href.replace("p-", "")
                if "section-" in href:
                    href = href.split("section-")[1]
                href = href.replace("§", "").replace(" ", "")
                href = href.replace("45 CFR", "").replace(" ","")
                if G.has_node(href):
                    # print(e_id, href)
                    if not G.has_edge(e_id, href):
                        G.add_edge(e_id, href, relation = "refer")
                        G.add_edge(href, e_id, relation = "referencedBy")
    return G


if __name__ == "__main__":
    '''
    Generate HIPAA graph
    '''
    
    input_files = ['subchapter-C_part-160.html', 'subchapter-C_part-162.html', 'subchapter-C_part-164.html']
    output_graph_file = "HIPAA.graphml"
    nodes = ["Part160","Part162","Part164"]
    generate_graph_from_htmls(input_files, output_graph_file, nodes)
    
    '''
    Generate HIPAA privacy rule graph
    The HIPAA privacy rule is located at 45 CFR Part 160 and Subparts A and E of Part 164.
    '''
    input_files = ['subchapter-C_part-160.html', 'subchapter-C_part-164.html']
    output_graph_file = "HIPAA_Privacy_Rule.graphml"
    nodes = ["Part160","Part164"]
    generate_hpr_graph_from_htmls(input_files, output_graph_file, nodes)


