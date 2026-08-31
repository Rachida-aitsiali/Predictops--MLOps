# Deploiement sur Komodo (infrastructure du cours)

Le professeur a fourni une plateforme partagee **Komodo** (self-hosted,
https://github.com/moghtech/komodo) plutot qu'un cloud public. Notre groupe
est **predictops**, avec un Stack dedie et une plage de ports **5200-5299**
sur le serveur partage.

- **UI Komodo (gestion)** : https://komodo.s3.fsbm.ma/login
- **Serveur d'execution (acces aux services deployes)** : `exp.s3.fsbm.ma:<port>`
- **Permissions du groupe sur son Stack** : Write, Inspect, Logs, Terminal
  (pas d'acces admin sur le Server partage, juste Read pour consulter ses
  infos).

## Comment fonctionne Komodo (en bref)

Komodo a deux composants :
- **Core** : l'interface web (ce que tu vois sur `komodo.s3.fsbm.ma`).
- **Periphery** : un agent installe sur le serveur partage, qui execute
  reellement `docker compose up` pour le compte de Core.

Un **Stack** dans Komodo correspond a un `docker-compose.yml`. Sa source peut
etre :
1. Un depot Git (recommande ici, car notre `docker-compose.yml` utilise
   `build: .` -> Periphery doit avoir acces au `Dockerfile` et au code) ;
2. Ou le contenu du compose colle directement dans l'UI (mais dans ce cas il
   faudrait une image deja publiee sur un registre, pas un `build` local).

**On utilisera l'option Git repo**, avec notre dossier `api/` comme
sous-dossier a builder.

## Etapes

### 1. Avoir un depot Git accessible

Le code de l'API doit etre dans un depot Git (GitHub par ex.) que le serveur
Komodo peut cloner. Si l'equipe n'a pas encore de depot commun, il faut en
creer un et y pousser ce projet (au moins le dossier `api/`).

### 2. Se connecter a Komodo

Va sur https://komodo.s3.fsbm.ma/login avec l'identifiant/mot de passe donne
lors de l'inscription au module. Ouvre le Stack du groupe **predictops**
(deja cree par l'administrateur d'apres le message du prof).

### 3. Configurer la source du Stack

Dans l'onglet **Configuration** du Stack :
- **Repo** : URL du depot Git de l'equipe (+ branche, ex. `main`).
- **Run directory / Path** : `api` (le sous-dossier ou se trouve
  `docker-compose.yml`).
- **Poll for updates / Webhook** (si disponible) : active pour redeployer
  automatiquement a chaque push.

### 4. Definir les variables d'environnement

Dans l'onglet **Environment** du Stack, ajouter (texte au format `.env`) :

```
API_PORT=5200
MODEL_PATH=/app/models/best_model.joblib
HISTORY_SIZE=15
```

`API_PORT` doit rester dans la plage assignee au groupe (**5200-5299**). Si
d'autres services de l'equipe (MLflow, Dagster, etc.) sont aussi deployes sur
ce Stack/serveur, coordonner avec l'equipe pour eviter que deux services
prennent le meme port (proposition : 5200 = API, a confirmer avec les
autres).

### 5. Deployer

Bouton **Deploy** (ou **Update**) dans l'UI Komodo. Periphery va cloner le
repo, executer `docker compose up --build -d` dans `api/`, et exposer le
port configure sur le serveur partage.

### 6. Verifier

- Onglet **Logs** du Stack dans Komodo pour voir les logs `uvicorn`.
- Depuis un navigateur ou `curl` :

  ```bash
  curl http://exp.s3.fsbm.ma:5200/health
  curl -X POST http://exp.s3.fsbm.ma:5200/predict \
    -H "Content-Type: application/json" \
    -d '{"machine_id":39,"temperature":85.2,"vibration":4.1,"humidity":55,"pressure":102,"energy_consumption":20}'
  ```

- L'onglet **Terminal** permet, si besoin, d'ouvrir un shell dans le
  conteneur pour debugger directement.

### 7. Mettre a jour le service

Un nouveau push sur le depot Git (branche configuree) redeploie
automatiquement si le webhook/poll est active ; sinon, cliquer de nouveau sur
**Deploy** dans l'UI.

## A confirmer avec l'equipe / le prof

- Nom exact du Stack predictops dans Komodo (deja cree ou a creer ?).
- Depot Git commun a utiliser comme source (URL).
- Port exact a utiliser dans 5200-5299 pour l'API (coordination avec les
  autres services de l'equipe).
