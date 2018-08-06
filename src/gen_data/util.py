import os
import numpy as np
import xml.etree.ElementTree as ET

from Box2D import b2ContactListener

from ..xml_writing.b2d_2_xml import XMLExporter
from ..sph.grids import GridManager, grids_from_dataframes
from ..sph.dataframes import dataframes_from_xml


# B2ContactListener for recording contacts and impulses
class ContactListener(b2ContactListener):
    def __init__(self, exporter: XMLExporter):
        super(ContactListener, self).__init__()

        self.xml_exp = exporter
        self.reset()

    # Reset the counter in preparation for a new step
    def reset(self):
        self.counter = 0

    # Store all pre-solve contact information
    def PreSolve(self, contact, _):
        # We give the contact an index so that we can recognize it later
        contact.userData = self.counter
        self.counter += contact.manifold.pointCount

        self.xml_exp.snapshot_contact(contact)

    # Store post-solve impulses
    def PostSolve(self, contact, impulse):
        self.xml_exp.snapshot_impulse(contact, impulse)


# Function for loading xml dataset and returning grids
def load_xml_return_grid(
        path, number, steps,
        body_channels, contact_channels,
        feature_channels, label_channels,
        p_ll, p_ur, xRes, yRes, h
):

    # If path is not absolute we make it
    if not os.path.isabs(path):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(file_dir, path)
    path += str(number) + "/"

    # We create the grid manager
    G = GridManager(p_ll, p_ur, xRes, yRes, h)

    features = []
    labels = []
    for i in range(1, steps+1):
        # We load the xml file
        try:
            filename = str(number) + "_" + str(i) + ".xml"
            xml = ET.ElementTree(file=path+filename).getroot()
        except:
            continue

        # We transfer the xml data onto grids
        df_b, df_c = dataframes_from_xml(xml)

        # We generate the grids
        fs, ls = grids_from_dataframes(
            G, df_b, df_c,
            body_channels,
            contact_channels,
            feature_channels,
            label_channels,
        )

        # We add the data
        features.append(fs)
        labels.append(ls)

    # We convert the data to a numpy array
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    return (features, labels)


# A function for loading grids saved as npz files using numpy
def load_grid(path, number):
    # If path is not absolute we make it
    if not os.path.exists(path):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(file_dir, path)

    filename = str(number) + ".npz"

    npzfile = np.load(os.path.join(path, filename))
    features = npzfile["features"]
    labels = npzfile["labels"]

    return (features, labels)
