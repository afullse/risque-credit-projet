"""Alternative Locust : locust -f scripts/locustfile.py --host http://127.0.0.1:8000
   --users 50 --spawn-rate 5 --run-time 2m --headless --csv reports/locust"""
from locust import HttpUser, between, task


class AgentCredit(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        self.client.post("/predict", json={"age": 35, "revenu_mensuel": 250000,
                                           "montant_credit": 1000000, "duree_mois": 12,
                                           "anciennete_activite": 4})

    @task(1)
    def health(self):
        self.client.get("/health")
