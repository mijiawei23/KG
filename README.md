# KG
This project is the code used to construct the industrial knowledge graph of China, Vietnam and Thailand.The project is divided into four parts, namely data set, knowledge extraction, knowledge purification and knowledge fusion. As shown in the code directory of the project
## Code directory
```bash
.
├── datasets/           # Used for storing datasets
├── extraction/         # Entity relationship extraction
├── purification/       # Purify the initially extracted entities and triples
└── fusion/             # Integrate knowledge of different languages
```
## Usage
We construct the knowledge graph by calling the deepseek large model, so it is necessary to apply for the api_key on [deepseek](https://www.deepseek.com/) for experimental use.
```bash
api_key="Your api_key"
model_name="model_name"
```

## Knowledge graph representation learning
In the experiment of knowledge graph representation learning, we used the [OpenKE](https://github.com/thunlp/OpenKE) tool to implement the experiment of link prediction, and we conducted experiments on models such as TransE respectively.
