
# Deployment Cheat‑Sheet (AWS)

If you clone this repo on an **AWS EC2 (Ubuntu)** box, these are the only commands you need.

## 1. System setup

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git tesseract-ocr
```

## 2. Clone the project

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>      # this folder should contain main.py, services/, requirements.txt
```

## 3. Python env + deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Env vars (create `.env`)

```bash
cat > .env << 'EOF'
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_MODEL=gpt-4o

RXCS_AI_TOKEN=some_secret_token
EOF
```

Adjust values as needed (or use `OPENAI_API_KEY` / `OPENAI_MODEL` instead of Azure).

## 5. Run the API (simple)

```bash
source .venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

Now:

- API: `http://<EC2_PUBLIC_IP>:8001`
- Docs: `http://<EC2_PUBLIC_IP>:8001/docs`

Call endpoints with header:

```http
Authorization: Bearer <RXCS_AI_TOKEN>
```

## 6. Run as a service (optional, production‑ish)

Create a systemd unit:

```bash
sudo tee /etc/systemd/system/ocr-api.service > /dev/null << 'EOF'
[Unit]
Description=OCR API (FastAPI + Uvicorn)
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/<your-repo>
Environment="PATH=/home/ubuntu/<your-repo>/.venv/bin"
ExecStart=/home/ubuntu/<your-repo>/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ocr-api
sudo systemctl start ocr-api
```

That’s it: the code in this repo will be running on AWS and restart automatically if the instance reboots.***

