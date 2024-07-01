from locust import HttpUser,task, between


videoPath = "C:\\Users\\pablo\\Desktop\\TFG (1)\\TFG (3).mp4"

class MyUser(HttpUser):
    wait_time = between(0,1) #Tiempo de espera entre peticiones en segundos

    @task(1)
    def get_cnn_structure(self):
        self.client.get('/api/model/structure/cnn')

    @task(2)
    def get_rnn_structure(self):
        self.client.get('/api/model/structure/rnn')
    
    @task(3)
    def get_cnn_graphs(self):
        self.client.get('/api/model/graphs/cnn')
    
    @task(4)
    def get_rnn_graphs(self):
        self.client.get('/api/model/graphs/rnn')

    @task(5)
    def get_cnn_confussion_matrix(self):
        self.client.get('/api/model/confussion/matrix/cnn')
    
    @task(6)
    def get_rnn_confussion_matrix(self):
        self.client.get('/api/model/confussion/matrix/rnn')
    
    @task(7)
    def predict_cnn(self):
        self.client.post('/api/predict',files={'video':open(videoPath,'rb')})

    @task(8)
    def predict_rnn(self):
        self.client.post('/api/predict/sequences',files={'video':open(videoPath,'rb')})
    