import os.path as osp

import torch
from torch_geometric.datasets import Reddit

class RawReddit(Reddit):
    def __init__(self, root="./", transform=None):
        super(RawReddit, self).__init__(root, transform=transform)

        self.name = "reddit"
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
        return "Post Title and Body"
    
    @property
    def domain(self):
        return "Online Communities"
    
    @property
    def raw_texts(self):
        return self._raw_data.raw_texts
    
    @property
    def category_names(self):
        return [
            "buildapc",
            "teenagers",
            "sex",
            "friendsafari",
            "TwoXChromosomes",
            "nba",
            "pokemon",
            "Android",
            "fantasyfootball",
            "starcraft",
            "relationships",
            "Random_Acts_Of_Amazon",
            "AskMen",
            "technology",
            "SquaredCircle",
            "DestinyTheGame",
            "IAmA",
            "hockey",
            "pcmasterrace",
            "explainlikeimfive",
            "guns",
            "gonewild",
            "aww",
            "trees",
            "fffffffuuuuuuuuuuuu",
            "Games",
            "Fitness",
            "news",
            "pokemontrades",
            "Minecraft",
            "gifs",
            "soccer",
            "science",
            "nfl",
            "CFB",
            "magicTCG",
            "atheism",
            "movies",
            "DotA2",
            "Music",
            "wow"

        ]

    @property
    def category_descriptions(self):
        buildqpc = "Planning on building a computer but need some advice? This is the place to ask! /r/buildapc is a community-driven subreddit dedicated to custom PC assembly. Anyone is welcome to seek the input of our helpful community as they piece together their desktop."
        teenagers = "r/teenagers is the biggest community forum run by teenagers for teenagers. Our subreddit is primarily for discussions and memes that an average teenager would enjoy to discuss about. We do not have any age-restriction in place but do keep in mind this is targeted for users between the ages of 13 to 19. Parents, teachers, and the like are welcomed to participate and ask any questions!"
        sex = "r/sex is for civil discussions pertaining to education and advice regarding your sexuality and sexual relationships. It is a sex-positive community and a safe space for people of all genders and orientations which demands respectful conduct in all exchanges. There is ZERO TOLERANCE FOR CREEPY OR HARASSING BEHAVIOR HERE — in posts, comments, messages, or any other contributions. No exceptions."
        friendsafari = "On April 8, 2024, Nintendo ended online support for 3DS. This subreddit is now an archive. ~~A place to exchange 3DS Friend Codes for the Pokémon X/Y Friend Safari!~~"
        TwoXChromosomes = "Welcome to TwoXChromosomes, a subreddit for both serious and silly content, and intended for women's perspectives. We are a welcoming subreddit and support the rights of all genders. Posts are moderated for respect, equanimity, grace, and relevance."
        nba = "A subreddit dedicated for NBA news and discussion."
        pokemon = "r/pokemon is an unofficial Pokémon fan community. This is the place for most things Pokémon on Reddit—TV shows, video games, toys, trading cards, you name it!"
        Android = "Android news, reviews, tips, and discussions about rooting, tutorials, and apps. General discussion about devices is welcome. Please direct technical support, upgrade questions, buy/sell, app recommendations, and carrier-related issues to other subreddits."
        fantasyfootball = "The biggest and best place on the internet for fantasy football discussion, strategy, and advice. Home of AMAugust where the biggest names in the industry answer your questions."
        starcraft = "All about the StarCraft games and professional scenes surrounding them. Please read the rules before submitting content."
        relationships = "/r/Relationships is a community built around helping people and the goal of providing a platform for interpersonal relationship advice between redditors. We seek posts from users who have specific and personal relationship quandaries that other redditors can help them try to solve."
        Random_Acts_Of_Amazon = "Amazon Wishlist Subreddit! Community, friends, gifting and fun! Random Acts with an Amazon Wishlist. Gift, get gifted, be merry, and have fun. We are NOT a needs-based subreddit."
        AskMen = "We don't read the rules, but we'll make a post anyway."
        technology = "Subreddit dedicated to the news and discussions about the creation and use of technology and its surrounding issues."
        SquaredCircle = "Reddit's largest professional wrestling community!"
        DestinyTheGame = "Welcome to Destiny Reddit! This sub is for discussing Bungie's Destiny 2 and its predecessor, Destiny. Please read the sidebar rules and be sure to search for your question before posting."
        IAmA = "I Am A, where the mundane becomes fascinating and the outrageous suddenly seems normal."
        hockey = "hockey: the best game on earth. Discuss the NHL, PWHL, IIHF, and all other hockey you can think of! We are the premier subreddit to talk everything hockey!"
        pcmasterrace = "PC Master Race - PCMR: A place where all enthusiasts of PC, PC gaming and PC technology are welcome! Welcome to the official subreddit of the PC Master Race / PCMR! All PC-related content is welcome, including build help, tech support, and any doubt one might have about PC ownership. You don't necessarily need a PC to be a member of the PCMR. You just have to love PCs. It's not about the hardware in your rig, but the software in your heart! Join us in celebrating and promoting tech, knowledge, and the best gaming, study, and work platform there exists. The Personal Computer."
        explainlikeimfive = "Explain Like I'm Five is the best forum and archive on the internet for layperson-friendly explanations. Don't Panic!"
        guns = "/r/guns: Firearms and related articles"
        gonewild = "Gonewild is a place for open-minded Adult Redditors to exchange their nude bodies for karma; showing it off in a comfortable environment without pressure."
        aww = "A subreddit for cute and cuddly pictures. Things that make you go AWW! -- like puppies, bunnies, babies, and so on... Feel free to post original pictures and videos of cute things."
        trees = "The go-to subreddit for anything and everything cannabis. From MMJ to munchies, from nugs to news, and everything between! The casual cannabis community."
        fffffffuuuuuuuuuuuu = "Rage Comics!"
        Games = "The goal of /r/Games is to provide a place for informative and interesting gaming content and discussions. Submissions should be for the purpose of informing or initiating a discussion, not just with the goal of entertaining viewers. Memes, comics, funny screenshots, arts-and-crafts, etc. will be removed."
        Fitness = "A place for the pursuit of physical fitness goals. Please see the r/Fitness Wiki and FAQ at https://thefitness.wiki for help with common questions."
        news = "The place for news articles about current events in the United States and the rest of the world. Discuss it all here."
        pokemontrades = "/r/pokemontrades is a trading community focusing on legitimate Pokémon. We are one of the few large Pokémon trading communities with a policy of no hacks, no clones!"
        Minecraft = "Minecraft community on Reddit."
        gifs = "GIFS"
        soccer = "The football subreddit. News, results, and discussion about the beautiful game."
        science = "This community is a place to share and discuss new scientific research. Read about the latest advances in astronomy, biology, medicine, physics, social science, and more. Find and submit new publications and popular science coverage of current research."
        nfl = "NFL: National Football League Discussion. The place to discuss all NFL related things"
        CFB = "The home of college football on reddit."
        magicTCG = "A diverse community of players devoted to Magic: the Gathering, a trading card game ('TCG') produced by Wizards of the Coast and originally designed by Richard Garfield. Join us discussing news, tournaments, gameplay, deckbuilding, strategy, lore, fan art, and more."
        atheism = "Welcome to r/atheism, the web's largest atheist forum. All topics related to atheism, agnosticism and secular living are welcome."
        movies = "The goal of /r/Movies is to provide an inclusive place for discussions and news about films with major releases. Submissions should be for the purpose of informing or initiating a discussion, not just to entertain readers. Read our extensive list of rules for more information on other types of posts like fan-art and self-promotion, or message the moderators if you have any questions."
        DotA2 = "/r/DotA2 is the most popular English-speaking community to discuss gameplay, esports, and news related to Valve's award winning free-to-play MOBA DotA 2."
        Music = "The musical community of reddit."
        wow = "World of Warcraft on Reddit!"
        return {
            "buildapc": buildqpc,
            "teenagers": teenagers,
            "sex": sex,
            "friendsafari": friendsafari,
            "TwoXChromosomes": TwoXChromosomes,
            "nba": nba,
            "pokemon": pokemon,
            "Android": Android,
            "fantasyfootball": fantasyfootball,
            "starcraft": starcraft,
            "relationships": relationships,
            "Random_Acts_Of_Amazon": Random_Acts_Of_Amazon,
            "AskMen": AskMen,
            "technology": technology,
            "SquaredCircle": SquaredCircle,
            "DestinyTheGame": DestinyTheGame,
            "IAmA": IAmA,
            "hockey": hockey,
            "pcmasterrace": pcmasterrace,
            "explainlikeimfive": explainlikeimfive,
            "guns": guns,
            "gonewild": gonewild,
            "aww": aww,
            "trees": trees,
            "fffffffuuuuuuuuuuuu": fffffffuuuuuuuuuuuu,
            "Games": Games,
            "Fitness": Fitness,
            "news": news,
            "pokemontrades": pokemontrades,
            "Minecraft": Minecraft,
            "gifs": gifs,
            "soccer": soccer,
            "science": science,
            "nfl": nfl,
            "CFB": CFB,
            "magicTCG": magicTCG,
            "atheism": atheism,
            "movies": movies,
            "DotA2": DotA2,
            "Music": Music,
            "wow": wow
        }