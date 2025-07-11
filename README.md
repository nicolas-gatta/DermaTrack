# DermaTrack
The aim of this application is to allow dermatological monitoring of patients.


## Table of contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Start Django Server](#start-the-django-server)
4. [User Login](#user-login-credentials)
5. [Usefull commands](#usefull-commands)

---

### Features

- Patient and visit management
- Image encryption and storage
- Super-resolution (SR) models to improve image quality
- Image preview generation
- Annotations on the image

---
### Prerequisites

#### 1. Install Python
Install ```python-3.11.11```. Follow the steps from the below reference document based on your Operating System.

Reference: [Python Version 3.11.11](https://www.python.org/downloads/release/python-31111/)

#### 2. Clone git repository or Download the ZIP
```bash
git clone "https://github.com/nicolas-gatta/DermaTrack.git"
```

#### 3. Download The .env (if not present)
Download the [.env](https://github.com/nicolas-gatta/DermaTrack/releases/download/V1.0.0/default.env) file, rename it ```.env``` and place it  in ```derma_track_src```

#### 4. Dataset (optional)
If you want to try and train the model, you have to download the dataset for the [training](https://github.com/nicolas-gatta/DermaTrack/releases/download/V1.0.0/training.zip), [validation](https://github.com/nicolas-gatta/DermaTrack/releases/download/V1.0.0/validation.zip) and [evaluation](https://github.com/nicolas-gatta/DermaTrack/releases/download/V1.0.0/evaluation.zip), you have to download them, unzip them and place them in ```derma_track_src\super_resolution\base_datasets```

#### 5. Go in the folder derma_track_src (if not done yet)
```bash
cd derma_track_src
```

#### 6. Install requirements
```bash
pip install -r requirements.txt
```

#### 7. Initialize the Databse
```bash
python manage.py migrate
```

#### 8. Fill the database with the initial Data
```bash
python manage.py loaddata data.json
```

---
### Start the Django Server

#### 1. Go in the folder derma_track_src (if not done yet)
```bash
cd derma_track_src
```

#### 2. Start the Django Server
```bash
python manage.py runserver
```
---
## User Login Credentials

Below are the credentials for the available user accounts:

 **Username**       | **Password** | **Role**     |
--------------------| ------------ | ------------ |
admin               | 1234         | Administator |
alijoh              | 1234         | Doctor       |
jansmi              | 1234         | Doctor       |
johdoe              | 1234         | Doctor       |
mabrow              | 1234         | Doctor       |
sarwil              | 1234         | Doctor       |
jamcla              | 1234         | Doctor       |
lismar              | 1234         | Doctor       |
davlee              | 1234         | Doctor       |
emmtho              | 1234         | Doctor       |
mikand              | 1234         | Doctor       |
rebdav              | 1234         | Doctor       |
josmoo              | 1234         | Doctor       |
annwal              | 1234         | Doctor       |

---
## Usefull Commands

```bash
python manage.py dumpdata --exclude admin --exclude contenttypes --exclude sessions --indent 2 > data.json
``` 