# Chapitre 8 — Deploiement FastAPI, Docker et CI/CD

**Redacteur : Israe — Role : DevOps / API Engineer**

## 8.1 Objectif du deploiement

Transformer le modele de prediction de pannes (entraine par Maroua, suivi via
MLflow par Rachida) en un service utilisable par une application tierce ou un
outil de supervision industrielle. L'objectif est d'exposer une API REST
stable, testee automatiquement, conteneurisee et prete a etre deployee sur
l'infrastructure cloud fournie par l'equipe pedagogique.

## 8.2 Architecture de l'API FastAPI

```
api/
├── app/
│   ├── main.py            # routes /health et /predict
│   ├── model_service.py    # chargement du modele + feature engineering en ligne
│   ├── schemas.py          # contrats de donnees (Pydantic)
│   └── config.py           # configuration (variables d'environnement)
├── models/best_model.joblib
├── tests/test_api.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Le modele sauvegarde par Maroua est un dictionnaire joblib contenant : le
pipeline scikit-learn complet (pretraitement + classifieur `GradientBoosting`),
le seuil de decision optimal (`0.18`, choisi pour maximiser le F1-score sous
contrainte de precision), et la liste exacte des 34 colonnes attendues en
entree.

Difficulte technique rencontree : le modele a ete entraine avec
`scikit-learn==1.6.1`. Le charger avec une version plus recente (1.9.0)
provoque une erreur de deserialisation (`ModuleNotFoundError: No module named
'_loss'`) a cause d'une reorganisation interne de scikit-learn. Le fichier
`requirements.txt` de l'API fixe donc precisement les memes versions que
celles utilisees a l'entrainement (`scikit-learn==1.6.1`, `pandas==2.2.3`,
`numpy==2.2.5`, `scipy==1.15.2`), garantissant la reproductibilite du service.

Autre point d'attention : le modele attend, en plus des 5 valeurs capteurs
brutes, des features glissantes par machine (valeur precedente, difference,
moyenne/ecart-type mobiles sur 5 et 15 releves). Une seule requete HTTP ne
contient pas cet historique. `model_service.py` maintient donc un petit
historique en memoire par `machine_id` (15 releves, configurable via
`HISTORY_SIZE`) et reproduit exactement les transformations de
`train_model.py::add_temporal_features` a chaque appel.

## 8.3 Endpoint GET /health

Verifie que le service est demarre et que le modele est bien charge en
memoire.

```json
{"status": "ok", "model_loaded": true, "model_type": "GradientBoostingClassifier"}
```

## 8.4 Endpoint POST /predict

Recoit un releve capteur d'une machine et retourne la probabilite de panne
ainsi qu'une decision binaire basee sur le seuil optimal du modele.

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

La validation des entrees (types, champs obligatoires) est geree par Pydantic :
une requete incomplete retourne automatiquement une erreur `422`.

## 8.5 Tests de l'API

Suite de tests `pytest` (`tests/test_api.py`) couvrant :

- `GET /health` retourne `status=ok` et `model_loaded=true` ;
- `POST /predict` retourne une prediction coherente (probabilite entre 0 et 1,
  label attendu) ;
- plusieurs appels successifs pour une meme machine (verifie que
  l'historique se construit sans erreur) ;
- une requete incomplete retourne bien un code `422`.

Resultat local : **4 tests passes**. L'API a egalement ete testee
manuellement via Swagger UI (`/docs`) et en ligne de commande (`curl`) : un
appel reel `POST /predict` execute depuis Swagger ("Try it out" > "Execute")
a renvoye une reponse `200` avec un corps coherent (ex. `panne_probable: true`,
`probabilite_panne: 0.2759`, seuil `0.18`), et `GET /health` a confirme
`model_loaded: true`.

## 8.6 Containerisation Docker

Le `Dockerfile` part d'une image `python:3.11-slim`, installe les dependances
figees, copie le code et le modele, puis lance `uvicorn` sur le port `8000`.
Un `HEALTHCHECK` interroge `/health` toutes les 30 secondes pour permettre a
l'orchestrateur (Docker, cloud) de detecter un service degrade.

Verifie : `docker build` termine sans erreur, le conteneur demarre et passe
`healthy` (`docker inspect` -> `"Status":"healthy"`), et les deux endpoints
repondent correctement depuis l'interieur du conteneur (`GET /health` ->
`model_loaded: true`, `POST /predict` -> reponse `200` coherente).

## 8.7 Docker Compose

`docker-compose.yml` construit l'image et demarre le conteneur avec les
variables d'environnement `MODEL_PATH` et `HISTORY_SIZE`, expose le port
`8000` et redemarre automatiquement le service en cas d'echec
(`restart: unless-stopped`).

Verifie : `docker compose up --build` construit l'image et demarre le
service, qui passe `healthy` et repond correctement sur `/health`.

## 8.8 Pipeline CI/CD avec GitHub Actions

Le workflow `.github/workflows/ci-cd.yml` se declenche a chaque push/PR
touchant le dossier `api/` et comprend deux jobs :

1. **test** : installation des dependances, lint du code (`ruff`), execution
   de la suite `pytest`.
2. **build** (dependant du succes des tests) : construction de l'image
   Docker, demarrage d'un conteneur ephemere et verification que `/health`
   repond correctement avant de valider le pipeline.

## 8.9 Deploiement cloud

L'infrastructure de deploiement est fournie par le professeur : une
plateforme self-hosted **Komodo** (https://komodo.s3.fsbm.ma), ou chaque
groupe dispose d'un **Stack** (le notre : **predictops**) avec les
permissions Write/Inspect/Logs/Terminal, et d'une plage de ports dediee
(**5200-5299**) sur le serveur partage (`exp.s3.fsbm.ma`).

Le service est deja compatible avec ce mode de deploiement : le port
publie par `docker-compose.yml` est parametrable via la variable
d'environnement `API_PORT` (defaut `5200`, dans la plage assignee), et
`MODEL_PATH`/`HISTORY_SIZE` restent configurables independamment. Komodo
deploie un Stack a partir d'un depot Git (clone + `docker compose up
--build`), ce qui correspond directement a notre `Dockerfile` +
`docker-compose.yml` sans etape de registre d'images intermediaire.

Le detail des etapes (connexion a Komodo, configuration du Stack, variables
d'environnement, verification) est documente dans
[`deploy/README-KOMODO.md`](deploy/README-KOMODO.md).

**Deploiement realise et verifie** : le Stack `predictops` est configure
avec le depot [`iris237111/predictops-mlops-api`](https://github.com/iris237111/predictops-mlops-api)
comme source, deployé sur le serveur `vh3` et exposé sur le port `5200`.
Difficulte rencontree : le premier build a echoue sur un timeout reseau pip
(`ReadTimeoutError` vers `files.pythonhosted.org`) lors du telechargement des
dependances sur le serveur partage ; corrige en ajoutant
`--default-timeout=120 --retries 5` a la commande `pip install` du
`Dockerfile`. Apres correction, le Stack est passe a l'etat **RUNNING** et
l'API repond publiquement :

- `GET http://exp.s3.fsbm.ma:5200/health` -> `{"status":"ok","model_loaded":true,"model_type":"GradientBoostingClassifier"}`
- Swagger UI accessible sur `http://exp.s3.fsbm.ma:5200/docs`

## 8.10 Monitoring du service

Le endpoint `/health` sert de sonde de disponibilite (liveness/readiness) et
est deja utilise par le `HEALTHCHECK` Docker et par le job `build` de la CI.
Il peut etre branche a tout systeme de supervision externe (ex. cron, uptime
monitor, ou le monitoring de derive prepare par Rachida) pour alerter en cas
d'indisponibilite du service ou de modele non charge.

## Captures a fournir

- Capture du code FastAPI (`app/main.py`, `app/model_service.py`).
- Capture de Swagger (`/docs`) avec les deux endpoints.
- Capture de l'appel `GET /health`.
- Capture de l'appel `POST /predict`.
- Capture du `Dockerfile`.
- Capture de `docker run` / `docker compose up` avec le conteneur demarre.
- Capture du workflow GitHub Actions (jobs `test` et `build` en succes).
- Capture du service deploye, si disponible.
