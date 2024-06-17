from lxml import etree
from bs4 import BeautifulSoup
import networkx as nx

#Final pathlength of the shortest path should divide by 15, since 15 pixels = 1 meter

svg_file_path = './HG_Floor_G_Annotated.svg'

def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'lxml')  

    text_div = soup.find(lambda tag: tag.name and tag.name.endswith('div') and 'font-size' in tag.get('style', ''))
    if text_div:
        font_tag = text_div.find('font')
        if font_tag:
            return font_tag.text.strip()
        else:
            return text_div.text.strip()
    
def extract_text_from_svg(svg_file_path):
    tree = etree.parse(svg_file_path)
    root = tree.getroot()

    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    foreign_objects = root.xpath('.//svg:foreignObject', namespaces=namespaces)

    nodes_and_edges_raw_form = []
    for fobj in foreign_objects:
        html_content = etree.tostring(fobj, encoding='utf-8').decode('utf-8')
        text = extract_text(html_content)

        box_info = {}
        current = fobj.getparent()
        while current is not None:
            prev = current.getprevious()
            rect = None
            while prev is not None:
                rect = prev.find('.//svg:rect', namespaces=namespaces)
                if rect is not None:
                    box_info['x'] = rect.get('x')
                    box_info['y'] = rect.get('y')
                    box_info['width'] = rect.get('width')
                    box_info['height'] = rect.get('height')
                    break
                prev = prev.getprevious()
            
            if rect is not None:
                break

            current = current.getparent()
        
        nodes_and_edges_raw_form.append({"text": text, "box_info": box_info})
    return nodes_and_edges_raw_form



nodes_and_edges_raw_form = extract_text_from_svg(svg_file_path)

G = nx.Graph()

for entry in nodes_and_edges_raw_form:
    text = entry['text']
    box_info = entry['box_info']
    node_type_prefix = text.split(':')[0]
    node_id = text.split(':')[1]

    if node_type_prefix in ['R', 'S', 'E']: 
        node_type = {'R': 'Room', 'S': 'Stairs', 'E': 'Elevator'}.get(node_type_prefix, 'Unknown')
        if node_type == 'Stairs':
            G.add_node(node_id, type=node_type, distance=float(box_info['height']))
        else:
            G.add_node(node_id, type=node_type, distance=float(box_info['width']))
    elif node_type_prefix in ['C', 'D']:
        node_ids = node_id.split('-')
        try:
            G.add_edge(node_ids[0], node_ids[1], distance=float(box_info['width']))
        except IndexError:
            print("C-D ERROR", entry)

print("Nodes of the graph:")
for node in G.nodes(data=True):
    print(node)

print("\nEdges of the graph:")
for edge in G.edges(data=True):
    print(edge)

