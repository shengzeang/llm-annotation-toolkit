import os.path as osp

import torch
from torch_geometric.datasets import WikiCS


class RawWikiCS(WikiCS):
    def __init__(self, root="./", transform=None):
        super(WikiCS, self).__init__(root, transform=transform)

        self.name = "wikics"
        self._raw_data = torch.load(osp.join(self.raw_dir, f"{self.name}.pt"))
        self._data.entity = self.entity
        self._data.domain = self.domain
        self._data.raw_texts = self.raw_texts
        self._data.category_names = self.category_names
        self._data.category_descriptions = self.category_descriptions

        self._data.x = self._raw_data.x
        self._data.y = self._raw_data.y
        self._data.edge_index = self._raw_data.edge_index
    
    @property
    def entity(self):
        return "Wiki interpretation of the word"
    
    @property
    def domain(self):
        return "Computer Science"
    
    @property
    def raw_texts(self):
        return self._raw_data.raw_texts
    
    @property
    def category_names(self):
        return ['Computational linguistics', 
                'Databases', 
                'Operating systems', 
                'Computer architecture', 
                'Computer security', 
                'Internet protocols', 
                'Computer file systems', 
                'Distributed computing architecture', 
                'Web technology', 
                'Programming language topics']
    
    @property
    def category_descriptions(self):
        Computational_linguistics = "Computational linguistics is an interdisciplinary field concerned with the statistical or rule-based modeling of natural language from a computational perspective."
        Databases = "A database is an organized collection of data, generally stored and accessed electronically from a computer system."
        Operating_systems = "An operating system (OS) is system software that manages computer hardware, software resources, and provides common services for computer programs."
        Computer_architecture = "Computer architecture is a set of rules and methods that describe the functionality, organization, and implementation of computer systems."
        Computer_security = "Computer security, cybersecurity, or information technology security is the protection of computer systems and networks from the theft of or damage to their hardware, software, or electronic data, as well as from the disruption or misdirection of the services they provide."
        Internet_protocols = "The Internet protocol suite is the conceptual model and set of communications protocols used in the Internet and similar computer networks."
        Computer_file_systems = "A file system is a method and data structure that the operating system uses to keep track of files on a disk or partition; that is, the way the files are organized on the disk."
        Distributed_computing_architecture = "Distributed computing is a field of computer science that studies distributed systems. A distributed system is a system whose components are located on different networked computers, which communicate and coordinate their actions by passing messages to one another."
        Web_technology = "Web technology refers to the means by which computers communicate with each other using markup languages and multimedia packages."
        Programming_language_topics = "A programming language is a formal language comprising a set of instructions that produce various kinds of output. Programming languages are used in computer programming to implement algorithms."

        return {
            'Computational linguistics': Computational_linguistics,
            'Databases': Databases,
            'Operating systems': Operating_systems,
            'Computer architecture': Computer_architecture,
            'Computer security': Computer_security,
            'Internet protocols': Internet_protocols,
            'Computer file systems': Computer_file_systems,
            'Distributed computing architecture': Distributed_computing_architecture,
            'Web technology': Web_technology,
            'Programming language topics': Programming_language_topics
        }