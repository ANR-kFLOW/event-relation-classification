from rdflib import PROV
# rdflib knows about quite a few popular namespaces, like W3C ontologies, schema.org etc.
import io
# rdflib knows about quite a few popular namespaces, like W3C ontologies, schema.org etc.
import uuid

import pydotplus
from IPython.display import display, Image
from rdflib import Literal, RDF, URIRef, XSD
from rdflib import Graph
from rdflib import PROV
# rdflib knows about quite a few popular namespaces, like W3C ontologies, schema.org etc.
from rdflib import RDFS
from rdflib.tools.rdf2dot import rdf2dot

base = 'http://kflow.eurecom.fr'
hong = {
    'name': 'hong',
    'title': 'Building a cross-document event-event relation corpus',
    'author': 'Hong, Yu and Zhang, Tongtao and O’Gorman, Tim and Horowit-Hendler, Sharone and Ji, Heng and Palmer, Martha',
    'booktitle': 'Proceedings of the 10th Linguistic Annotation Workshop held in conjunction with ACL 2016 (LAW-X 2016)',
    'year': '2016'
}



GPT3 ={
    'name': 'GPT3',
    'title': 'Language Models are Few-Shot Learners',
    'author': 'Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and Agarwal, Sandhini and Herbert-Voss, Ariel and Krueger, Gretchen and Henighan, Tom and Child, Rewon and Ramesh, Aditya and Ziegler, Daniel and Wu, Jeffrey and Winter, Clemens and Hesse, Chris and Chen, Mark and Sigler, Eric and Litwin, Mateusz and Gray, Scott and Chess, Benjamin and Clark, Jack and Berner, Christopher and McCandlish, Sam and Radford, Alec and Sutskever, Ilya and Amodei, Dario',
    'booktitle': 'Advances in Neural Information Processing Systems',
    'year': '2020'
}

OntoED = {
    'name': 'OntoED',
    'title': 'Ontoed: Low-resource event detection with ontology embedding',
    'author': 'Deng, Shumin and Zhang, Ningyu and Li, Luoqiu and Chen, Hui and Tou, Huaixiao and Chen, Mosha and Huang, Fei and Chen, Huajun',

    'year': '2021'
}
timebank = {
    'name': 'timebank',
    'title': 'The timebank corpus',
    'author': 'Pustejovsky, James and Hanks, Patrick and Sauri, Roser and See, Andrew and Gaizauskas, Robert and Setzer, Andrea and Radev, Dragomir and Sundheim, Beth and Day, David and Ferro, Lisa and others',

    'year': '2003'
}

faro = {
    'name': 'faro',
    'title': 'Beyond Causality: Representing Event Relations in Knowledge Graphs',
    'author': 'Youssra Rebboud, Pasquale Lisena, and Raphael Troncy',
    'year': '2022'
}
def bind(g,namespace,url):
    g.bind(namespace, url)


def node_creation(seed, entity_mention,type):

    uri = base +'/'+type.strip()+'/'+str(uuid.uuid5(uuid.NAMESPACE_DNS, seed + str(entity_mention)))
    return URIRef(uri)

def add_relation(g,node,predicate,obj):
    return (g.add((node, URIRef(predicate), obj)))

def add_provenance(g,statement,prov):
    g.add((Literal(statement),PROV.wasGeneratedBy, prov))

def add_label(g,a, label):
        g.add([a, RDFS.label, Literal(label)])

def visualize(g):
    stream = io.StringIO()
    rdf2dot(g, stream, opts = {display})
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    display(Image(png))


def add_type(g,node, typ):
    if typ == 'event':
        g.add((node, RDF.type, URIRef('http://purl.org/faro/Event')))
    if typ == 'condition':
        g.add((node, RDF.type, URIRef('http://purl.org/faro/Condition')))
    if typ == 'time':
        g.add((node, RDF.type, URIRef('https://www.w3.org/TR/owl-time/#time:TemporalEntity')))


def create_provenance_nodes(g,ref):
    name = ref['name']

    node_name = node_creation('', name + 'paper','paper')

    add_relation(g, node_name, 'http://purl.org/dc/elements/1.1/title', Literal(ref['title']))
    add_relation(g, node_name, 'http://purl.org/spar/fabio/hasPublicationYear', Literal(ref['year'], datatype=XSD.gYear))
    add_relation(g, node_name, 'http://purl.org/dc/terms/creator', Literal(ref['author']))
    return node_name





