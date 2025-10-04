# FarmLink Copilot

FarmLink est un assistant RAG dédié aux filières agricoles d'Afrique de l'Ouest. Il combine une interface Streamlit moderne et une API FastAPI reliée à Qdrant pour offrir des réponses contextualisées à partir de documents experts (politiques agricoles, gestion des sols, irrigation, mécanisation, etc.).

## Structure du projet

```
farmlink/
├── backend/      # FastAPI + ingestion Qdrant + retrievers
└── frontend/     # Interface Streamlit façon chatbot
```

## Pré-requis

- Python 3.10+
- Qdrant Cloud (ou self-hosted) et les clés API associées
- Clé API Mistral

## Installation

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt  # si environnement dédié
```

Créer un fichier `backend/.env` pour configurer Qdrant et Mistral (voir `backend/.env.example`).

## Ingestion des corpus

```bash
cd backend
python ingest/ingest_qdrant.py \
  --folder data/raw/marche \
  --collection farmlink_marche \
  --domain marche
```

Répéter pour chaque domaine (`sols`, `eau`, `meca`, `cultures`). Les chunks sont automatiquement taggés avec un label de source lisible.

## Lancement

### API FastAPI
```bash
cd backend
uvicorn app:app --reload --port 8000
```

### Interface Streamlit
```bash
cd frontend
streamlit run app.py
```

## Notes
- Le prompt LLM force un ratio 60 % documents FarmLink / 40 % contextualisation externe.
- Les salutations sont gérées côté backend pour un onboarding soigné.
- Les IDs de documents sont formatés pour être lisibles dans la liste des sources.

Bon usage et bon pitch pour le projet final !
