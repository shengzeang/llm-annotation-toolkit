import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset


class RawOGB(PygNodePropPredDataset):
    def __init__(self, name="ogbn-arxiv", root="./", transform=None):
        if name not in ["ogbn-arxiv"]:
            raise ValueError("Dataset name not supported!")
        super(RawOGB, self).__init__(name, root, transform=transform)

        self.name = name
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
        return "Paper abstract"
    
    @property
    def domain(self):
        if self.name == 'ogbn-arxiv':
            return "Computer Science"

    
    @property
    def raw_texts(self):
        return self._raw_data.raw_texts
    
    @property
    def category_names(self):
        if self.name == 'ogbn-arxiv':
            return [
        'Numerical Analysis',
        'Multimedia',
        'Logic in Computer Science',
        'Computers and Society',
        'Cryptography and Security',
        'Distributed, Parallel, and Cluster Computing',
        'Human-Computer Interaction',
        'Computational Engineering, Finance, and Science',
        'Networking and Internet Architecture',
        'Computational Complexity',
        'Artificial Intelligence',
        'Multiagent Systems',
        'General Literature',
        'Neural and Evolutionary Computing',
        'Symbolic Computation',
        'Hardware Architecture',
        'Computer Vision and Pattern Recognition',
        'Graphics',
        'Emerging Technologies',
        'Systems and Control',
        'Computational Geometry',
        'Other Computer Science',
        'Programming Languages',
        'Software Engineering',
        'Machine Learning',
        'Sound',
        'Social and Information Networks',
        'Robotics',
        'Information Theory',
        'Performance',
        'Computation and Language',
        'Information Retrieval',
        'Mathematical Software',
        'Formal Languages and Automata Theory',
        'Data Structures and Algorithms',
        'Operating Systems',
        'Computer Science and Game Theory',
        'Databases',
        'Digital Libraries',
        'Discrete Mathematics'
    ]

    
    @property
    def category_descriptions(self):
        if self.name == 'ogbn-arxiv':
            na = 'The study of algorithms for performing numerical computations. This includes solving equations, integration, differentiation, and other operations where numerical approximations are necessary.'
            mm = 'This field involves the processing, generation, manipulation, and analysis of multimedia content such as audio, video, images, and animations.'
            lo = 'Focuses on the application of logic to computer science, including areas like formal verification, automated reasoning, and logic programming.'
            cy = 'Examines the impact of computers on society, including ethical, legal, and social issues, as well as the ways society influences computing technology.'
            cr = 'The study of techniques for secure communication in the presence of third parties. This includes encryption, authentication, secure protocols, and related security measures.'
            dc = 'Focuses on systems that use multiple processors or computers to perform computations in parallel or distributed fashion, enhancing performance and reliability.'
            hc = 'The study of how people interact with computers and designing user interfaces that improve user experience and accessibility.'
            ce = 'Applies computational techniques to solve problems in engineering, finance, and science, involving simulations, optimizations, and modeling.'
            ni = 'Covers the design, implementation, and analysis of computer networks, including protocols, network architecture, and performance evaluation.'
            cc = 'Studies the complexity of algorithms and problems, classifying them according to the resources they require for their solution, such as time and space.'
            ai = 'The simulation of human intelligence in computers, encompassing areas like machine learning, natural language processing, robotics, and expert systems.'
            ma = 'Investigates systems composed of multiple interacting agents, which can be autonomous entities like robots or software programs.'
            gl = 'Encompasses broad and interdisciplinary studies within computer science that do not fit into other specific categories.'
            ne = 'Combines neural networks and evolutionary algorithms to solve complex optimization and learning problems.'
            sc = 'Involves the manipulation and solving of mathematical expressions symbolically, as opposed to numerical methods.'
            ar = 'The design and organization of computer hardware, including processors, memory systems, and input/output devices.'
            cv = 'Focuses on enabling computers to interpret and understand visual information from the world, recognizing patterns and objects.'
            gr = 'The study of generating and manipulating visual content, including rendering, animation, and visualization techniques.'
            et = 'Research on new and innovative technologies in the field of computer science, including quantum computing and nanotechnology.'
            sy = 'Covers the design and analysis of control systems and their applications in engineering and computing.'
            cg = 'Studies algorithms for solving geometric problems, such as those involving points, lines, and polygons in two or more dimensions.'
            oh = 'Includes topics in computer science that do not fall into other predefined categories.'
            pl = 'The design, implementation, and analysis of programming languages, including syntax, semantics, and type systems.'
            se = 'The application of engineering principles to software development, including methodologies, tools, and practices for designing, developing, and maintaining software.'
            lg = 'A subset of artificial intelligence focused on building systems that can learn from data and improve their performance over time without explicit programming.'
            sd = 'The study of sound processing, including audio signal processing, speech recognition, and music information retrieval.'
            si = 'Examines the structure and behavior of social and information networks, including graph theory, network analysis, and social network dynamics.'
            ro = 'The design, construction, operation, and application of robots, including areas like motion planning, perception, and human-robot interaction.'
            it = 'Studies the quantification, storage, and communication of information, including coding theory and data compression.'
            pf = 'The analysis and improvement of system performance, including benchmarking, performance modeling, and optimization.'
            cl = 'Focuses on natural language processing, enabling computers to understand, interpret, and generate human language.'
            ir = 'The study of obtaining relevant information from large collections of data, including search engines and data mining.'
            ms = 'The development and analysis of software for solving mathematical problems, including numerical libraries and symbolic computation tools.'
            fl = 'Studies abstract machines and formal languages, including automata theory, grammars, and the theory of computation.'
            ds = 'The design and analysis of data structures and algorithms for efficiently solving computational problems.'
            os = 'The design and implementation of operating systems, managing hardware resources, and providing services for computer programs.'
            gt = 'The application of game theory to computer science, including algorithmic game theory and the study of strategic interactions in computational settings.'
            db = 'The design, implementation, and management of databases, including data modeling, query languages, and database systems.'
            dl = 'The organization, management, and retrieval of digital information, including digital archiving and information access.'
            dm = 'The study of mathematical structures that are fundamentally discrete rather than continuous, including graph theory, combinatorics, and logic.'

            return {
                'Numerical Analysis': na,
                'Multimedia': mm,
                'Logic in Computer Science': lo,
                'Computers and Society': cy,
                'Cryptography and Security': cr,
                'Distributed, Parallel, and Cluster Computing': dc,
                'Human-Computer Interaction': hc,
                'Computational Engineering, Finance, and Science': ce,
                'Networking and Internet Architecture': ni,
                'Computational Complexity': cc,
                'Artificial Intelligence': ai,
                'Multiagent Systems': ma,
                'General Literature': gl,
                'Neural and Evolutionary Computing': ne,
                'Symbolic Computation': sc,
                'Hardware Architecture': ar,
                'Computer Vision and Pattern Recognition': cv,
                'Graphics': gr,
                'Emerging Technologies': et,
                'Systems and Control': sy,
                'Computational Geometry': cg,
                'Other Computer Science': oh,
                'Programming Languages': pl,
                'Software Engineering': se,
                'Machine Learning': lg,
                'Sound': sd,
                'Social and Information Networks': si,
                'Robotics': ro,
                'Information Theory': it,
                'Performance': pf,
                'Computation and Language': cl,
                'Information Retrieval': ir,
                'Mathematical Software': ms,
                'Formal Languages and Automata Theory': fl,
                'Data Structures and Algorithms': ds,
                'Operating Systems': os,
                'Computer Science and Game Theory': gt,
                'Databases': db,
                'Digital Libraries': dl,
                'Discrete Mathematics': dm
            }
        