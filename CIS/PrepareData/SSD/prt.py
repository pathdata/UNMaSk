from xml.etree import ElementTree
from xml.dom import minidom
#import lxml.etree as etree
import xml.etree.cElementTree as ET
from xml.dom.minidom import parse, parseString
import xml.dom.minidom
#import lxml.etree
#import lxml.etree as etree
#from xml.etree.ElementTree import ElementTree


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    #x = tree.parse("figfnal.xml")
    return reparsed.toprettyxml(indent="  ")
