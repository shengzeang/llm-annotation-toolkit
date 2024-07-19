import os.path as osp

import torch
from torch_geometric.datasets import Planetoid


class RawPlanetoid(Planetoid):
    def __init__(self, root="./", name="Cora", split="public", transform=None):
        if name not in ["Cora", "Citeseer", "Pubmed"]:
            raise ValueError("Dataset name not supported!")
        super(RawPlanetoid, self).__init__(root, name, split, transform=transform)

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
        if self.name == 'Pubmed':
            return "Medicine"
        elif self.name in ['Cora', 'Citeseer']:
            return "Computer Science"
    
    @property
    def raw_texts(self):
        return self._raw_data.raw_texts
    
    @property
    def category_names(self):
        if self.name == 'Cora':
            return ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Theory", "Reinforcement Learning", "Probabilistic Methods"]
        elif self.name == 'Citeseer':
            return ["Agents", "Machine Learning", "Information Retrieval", "Databases", "Human-Computer Interaction", "Artificial Intelligence"]
        elif self.name == 'Pubmed':
            return ["Diabetes Mellitus, Experimental", "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"]
    
    @property
    def category_descriptions(self):
        if self.name == 'Cora':
            Rule_Learning_dec = "Rule learning in arXiv's Computer Science (CS) subcategories refers to the process of automatically extracting useful rules or patterns from data. This approach is commonly used in machine learning and data mining tasks. Rule learning algorithms aim to discover understandable and interpretable rules that capture relationships between variables or features in a dataset. These rules can then be applied to make predictions, classify data, or gain insights into the underlying structure of the data. Rule learning techniques are utilized across various CS subfields, including machine learning, data mining, artificial intelligence, and knowledge representation. Researchers develop and refine rule learning algorithms to address challenges such as scalability, interpretability, and generalization to diverse datasets. The application areas of rule learning span from business analytics and finance to healthcare and natural language processing, highlighting its versatility and significance in computational research."
            Neural_Networks_dec = "Neural networks, a cornerstone of modern artificial intelligence, feature prominently in arXiv's Computer Science (CS) subcategories. These networks consist of interconnected nodes (neurons) organized in layers, where each neuron processes and transmits information. Within arXiv's CS subcategories, neural networks are extensively researched and applied across various domains, including machine learning, computer vision, natural language processing, and robotics.Researchers explore novel architectures, training algorithms, and optimization techniques to improve the performance and efficiency of neural networks. Convolutional Neural Networks (CNNs) excel in image recognition tasks, while Recurrent Neural Networks (RNNs) are adept at processing sequential data like text or time series. Additionally, attention mechanisms, transformers, and graph neural networks have emerged as powerful tools for tasks involving complex relationships and structures."
            Case_Based_dec = "In arXiv's Case Based subcategories, agents refer to intelligent entities capable of reasoning, learning, and decision-making within dynamic environments. These agents interact with their surroundings to achieve specific goals or solve problems, often leveraging past experiences stored as cases to inform their actions. Case-based reasoning (CBR) systems, prevalent in this domain, enable agents to adapt and solve new problems by retrieving and reusing similar past solutions.Agents in Case Based subcategories exhibit various characteristics, including adaptability, learning from experience, and the ability to generalize solutions across different contexts. They employ techniques such as similarity assessment, case adaptation, and knowledge representation to effectively utilize past experiences in current decision-making processes. These agents find applications in diverse domains such as autonomous systems, medical diagnosis, recommender systems, and fault diagnosis, where the ability to draw upon past experiences is invaluable for making informed decisions in complex and uncertain environments."
            Genetic_Algorithms_dec = "Genetic Algorithms (GAs) are optimization techniques inspired by the principles of natural selection and genetics. Within arXiv's Computer Science (CS) subcategories, GAs typically fall under the category of Evolutionary Computation.In a GA, a population of candidate solutions (often represented as chromosomes or individuals) undergoes a process of evolution through successive generations. Each individual in the population represents a potential solution to the optimization problem at hand. Through the iterative process of selection, crossover, and mutation, individuals with favorable traits are selected to produce offspring, which inherit characteristics from their parent solutions. Over time, this iterative process tends to improve the quality of solutions within the population."
            Theory_dec = "In the arXiv Computer Science (CS) subcategories, Theory typically refers to the foundational aspects of computer science, including the mathematical frameworks, algorithms, complexity theory, cryptography, and formal methods. It encompasses research into the fundamental principles that underlie computation and information processing. This area often involves proving theorems, analyzing algorithms, and exploring the limits of what can be computed efficiently or at all."
            Reinforcement_Learning_dec = "In arXiv's computer science subcategories, reinforcement learning (RL) is a dynamic field of research focused on teaching agents to make sequential decisions through interaction with environments. Here's a brief overview across some of these subcategories:Machine Learning (cs.LG): RL algorithms are developed and studied within this category, focusing on learning policies to maximize cumulative rewards.Artificial Intelligence (cs.AI): RL techniques are explored here for building intelligent systems capable of decision-making in complex environments.Robotics (cs.RO): RL is utilized for training robots to perform tasks autonomously by learning from interactions with their surroundings.Computer Vision and Pattern Recognition (cs.CV): RL methods are applied to tasks like object recognition, scene understanding, and image generation to enhance learning-based vision systems"
            Probabilistic_Methods_dec = "Probabilistic methods in computer science, as explored across various arXiv subcategories, involve techniques that leverage probability theory to model uncertainty and make decisions under uncertainty. These methods are widely applied in machine learning, artificial intelligence, robotics, computer vision, and other fields. In machine learning (cs.LG), probabilistic methods include Bayesian inference, probabilistic graphical models, and probabilistic programming, which are used for tasks such as classification, regression, and clustering while accounting for uncertainty in data and model parameters.In artificial intelligence (cs.AI), probabilistic reasoning is utilized for decision-making in uncertain environments, such as in probabilistic graphical models for representing knowledge and making inferences in expert systems or for planning under uncertainty in autonomous agents. In robotics (cs.RO), probabilistic methods are crucial for localization, mapping, and perception tasks, where sensor measurements are noisy and environments are dynamic."
            return {"Rule Learning": Rule_Learning_dec, "Neural Networks": Neural_Networks_dec, "Case Based": Case_Based_dec,
                        "Genetic Algorithms": Genetic_Algorithms_dec, "Theory":Theory_dec, "Reinforcement Learning": Reinforcement_Learning_dec,
                        "Probabilistic Methods": Probabilistic_Methods_dec}
        
        elif self.name == 'Citeseer':
            Agents_dec="In arXiv's computer science subcategories, Agents typically refers to research involving autonomous agents, which are entities that can act independently and make decisions based on their environment and goals. These agents are often studied in fields such as artificial intelligence, multi-agent systems, robotics, and game theory. Research in this area may include topics like agent-based modeling, reinforcement learning, swarm intelligence, and negotiation protocols. The goal is to understand how individual agents can interact and cooperate within complex systems, leading to applications in areas like autonomous vehicles, smart grids, and decentralized networks."
            Machine_Learning_dec="In arXiv's computer science subcategories, Machine Learning encompasses a diverse range of research areas focused on developing algorithms and models that enable computers to learn from data and make predictions or decisions without explicit programming. This interdisciplinary field intersects with artificial intelligence, statistics, optimization, and data mining. Research in Machine Learning spans topics such as supervised learning, unsupervised learning, reinforcement learning, deep learning, probabilistic modeling, and kernel methods. Key applications include image and speech recognition, natural language processing, recommender systems, anomaly detection, and predictive analytics. The overarching goal of Machine Learning research is to advance the capabilities of intelligent systems, enabling them to autonomously acquire knowledge, adapt to new situations, and improve performance over time."
            Information_Retrieval_dec="Information Retrieval (IR) in arXiv's Computer Science (CS) subcategories involves the study of methods and techniques for efficiently and effectively accessing, organizing, and retrieving relevant information from large collections of data, particularly within the domain of computer science. This encompasses various subfields such as natural language processing, machine learning, databases, and information retrieval itself. Researchers in this area focus on developing algorithms, models, and systems for tasks like document indexing, query processing, relevance ranking, and user interaction to facilitate efficient access to relevant information from diverse sources."
            Databases_dec="Databases in arXiv's Computer Science (CS) subcategories encompass a broad range of research focusing on the theory, design, implementation, and optimization of data storage and management systems. This includes relational databases, NoSQL databases, distributed databases, and database management systems (DBMS). Researchers in this area explore various topics such as query processing and optimization, indexing techniques, transaction management, concurrency control, data modeling, database security, and scalability. They investigate innovative approaches to address the challenges of handling large-scale data, ensuring data consistency and integrity, improving query performance, and enabling efficient data access and retrieval. Additionally, advancements in machine learning and artificial intelligence are increasingly integrated into database systems to enhance data analytics, recommendation systems, and decision support applications. Overall, databases in arXiv's CS subcategories play a crucial role in advancing the state-of-the-art in data management technologies to meet the evolving needs of modern computing environments."
            Human_Computer_Interaction_dec="Human-Computer Interaction (HCI) in arXiv's Computer Science (CS) subcategories involves the study of the interaction between humans and computer systems, with the aim of designing intuitive, effective, and enjoyable user experiences. This interdisciplinary field draws upon principles from psychology, design, computer science, and other areas to understand how users interact with technology and how to design interfaces that meet their needs and preferences. Researchers in HCI explore topics such as usability testing, user interface design, interaction techniques, accessibility, user experience (UX) design, human factors, and cognitive psychology. They develop methodologies, tools, and frameworks to evaluate and enhance the usability, accessibility, and overall quality of interactive systems across various domains, including desktop applications, mobile devices, virtual reality (VR), augmented reality (AR), and wearable technologies. Additionally, HCI research often involves user-centered design approaches, iterative prototyping, and user feedback to ensure that computer systems are designed with the end-user in mind, ultimately aiming to create technology that enhances human capabilities and improves quality of life."
            Artificial_Intelligence_dec="Artificial Intelligence (AI) in arXiv's Computer Science (CS) subcategories covers a wide spectrum of research focusing on the development, application, and understanding of intelligent systems. This interdisciplinary field draws from various subfields such as machine learning, natural language processing, computer vision, robotics, and knowledge representation. Researchers in AI explore topics such as algorithm development, model training and optimization, AI ethics and fairness, human-AI interaction, and AI applications across different domains."
            return {"Agents": Agents_dec, "Machine Learning":Machine_Learning_dec, "Information Retrieval": Information_Retrieval_dec,
                            "Databases": Databases_dec, "Human-Computer Interaction": Human_Computer_Interaction_dec,
                            "Artificial Intelligence": Artificial_Intelligence_dec}
        
        elif self.name == 'Pubmed':
            Experimental_dec = "Diabetes Mellitus, Experimental refers to diabetes mellitus induced experimentally by administration of various diabetogenic agents or by pancreatectomy."
            Type1_dec = "Diabetes Mellitus Type 1 is a subtype of diabetes mellitus that is characterized by insulin deficiency. It is manifested by the sudden onset of severe hyperglycemia, rapid progression to diabetic ketoacidosis, and death unless treated with insulin. The disease may occur at any age, but is most common in childhood or adolescence."
            Type2_dec = "Diabetes Mellitus Type 2 is a subclass of diabetes mellitus that is not insulin-responsive or dependent (NIDDM). It is characterized initially by insulin resistance and hyperinsulinemia; and eventually by glucose intolerance; hyperglycemia; and overt diabetes. Type 2 diabetes mellitus is no longer considered a disease exclusively found in adults. Patients seldom develop ketosis but often exhibit obesity."
            return {"Diabetes Mellitus, Experimental": Experimental_dec, "Diabetes Mellitus Type 1": Type1_dec, "Diabetes Mellitus Type 2": Type2_dec}

