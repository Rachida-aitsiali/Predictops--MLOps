# API de maintenance predictive — Partie Israe (DevOps / API Engineer)

Expose le modele entraine par Maroua (tracking MLflow de Rachida) derriere une
API REST FastAPI, conteneurisee et testee automatiquement via GitHub Actions.

## Structure

```
api/
├── app/
│   ├── main.py            # points d'entree FastAPI (/health, /predict)
│   ├── model_service.py    # chargement du modele + feature engineering en ligne
│   ├── schemas.py          # schemas Pydantic (requete / reponse)
│   └── config.py           # configuration via variables d'environnement
├── models/
│   └── best_model.joblib   # modele fourni par Maroua (pipeline + seuil + features)
├── tests/
│   └── test_api.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── docker-compose.yml
```

## Pourquoi un historique par machine ?

Le modele a ete entraine avec des features glissantes par machine (moyenne et
ecart-type mobiles sur 5/15 releves, valeur precedente, difference). Une seule
requete `/predict` ne peut pas fournir ces features : l'API conserve donc un
petit historique en memoire (`HISTORY_SIZE=15` releves par `machine_id`,
configurable) et recalcule exactement les memes transformations que
`train_model.py::add_temporal_features`. Les toutes premieres requetes pour
une machine (historique vide) produisent des valeurs manquantes, imputees par
le `SimpleImputer` deja present dans le pipeline scikit-learn — comme pendant
l'entrainement.

Limite connue : l'historique est en memoire, donc reinitialise a chaque
redemarrage du conteneur. Une evolution possible serait de le persister
(Redis, DuckDB) si l'API est deployee avec plusieurs replicas.

## Lancer en local

```powershell
cd api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Swagger UI : http://localhost:8000/docs

## Lancer avec Docker

```powershell
cd api
docker compose up --build
```

## Endpoints

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "model_type": "GradientBoostingClassifier"}
```

### `POST /predict`

Requete :

```json
{
  "machine_id": 39,
  "temperature": 85.2,
  "vibration": 4.1,
  "humidity": 55,
  "pressure": 102,
  "energy_consumption": 20
}
```

Reponse :

```json
{
  "machine_id": 39,
  "panne_probable": false,
  "probabilite_panne": 0.1049,
  "seuil_utilise": 0.18,
  "label": "panne non probable"
}
```

## Tests

```powershell
cd api
pytest -v
```

## CI/CD

Le workflow [`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml)
s'execute a chaque push/PR touchant `api/` :

1. **test** : installation des dependances, lint (`ruff`), `pytest`.
2. **build** : build de l'image Docker puis verification du endpoint
   `/health` sur un conteneur ephemere.

## Deploiement (Komodo, infrastructure du cours)

L'infrastructure de deploiement est la plateforme **Komodo** fournie par le
professeur (https://komodo.s3.fsbm.ma), groupe **predictops**, plage de
ports **5200-5299** sur `exp.s3.fsbm.ma`. Le port publie par
`docker-compose.yml` est parametrable via `API_PORT` (voir `.env.example`).
Guide complet : [`deploy/README-KOMODO.md`](deploy/README-KOMODO.md).
