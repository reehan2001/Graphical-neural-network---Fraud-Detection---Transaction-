import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense , Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots



class GraphAutoencoder:
    def __init__(self,encoding_dim):
        self.encodings_dims = encoding_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    
    def build_autoencoder(self,num_nodes):
        input_layer = Input(shape=(num_nodes,))
        encoded = Dense(self.encodings_dims,activation='relu')(input_layer)

        self.encoder = Model(input_layer,encoded)
        encoded_input = Input(shape=(self.encodings_dims,))
        decoded = Dense(num_nodes,activation='sigmoid')(encoded_input)
        self.decoder = Model(encoded_input,decoded)

        autoencoder_output = self.decoder(self.encoder(input_layer))
        self.autoencoder = Model(input_layer,autoencoder_output)


    def compile_autoencoder(self):
        self.autoencoder.compile(optimizer='adam',loss='binary_crossentropy')

    def train_autoencoder(self,adjacency_matrix,epochs,batch_size):
        self.autoencoder.fit(adjacency_matrix,adjacency_matrix,epochs=epochs,batch_size=batch_size,shuffle=True)

    
    def encode(self,adjacency_matrix):
        encoded_data = self.encoder.predict(adjacency_matrix)
        return encoded_data
    
    def decode(self,encoded_data):
        decoded_data = self.decoder.predict(encoded_data)
        return decoded_data

class GraphVisualizer:
    def __init__(self):
        self.fig = None

    def visualizer_graph(self,adjacency_matrix,encoded_data,decoded_data):
        num_nodes = adjacency_matrix.shape[0]

        self.fig = make_subplots(rows=1,cols=3,subplot_titles=['Original Graph','Encoded Graph','Decoded Graph'])

        self.__add_edges(adjacency_matrix,row=1,col=1)
        self.__add_nodes(list(range(num_nodes)),list(range(num_nodes)),row=1,col=1)

        self.__add_edges(encoded_data,row=1,col=2)
        self.__add_nodes(encoded_data[:,0],encoded_data[:,1],row=1,col=2)

        self.__add_edges(decoded_data,row=1,col=3)
        self.__add_nodes(decoded_data[:,0],decoded_data[:,1],row=1,col=3)

        self.fig.update_layout(height=400,width=900,title_text="Graph Autoencoder Visulaization")
        self.fig.show()

    
    def __add_edges(self,adjacency_matrix,row,col):
        edges = np.where(adjacency_matrix == 1)
        edges_trace = go.Scatter(x=edges[1],y=edges[0],mode='lines',name="Edges",line=dict(color='black'))
        self.fig.add_trace(edges_trace,row=row,col=col)

    def __add_nodes(self,x,y,row,col):
        nodes_trace =  go.Scatter(x=x,y=y,mode='markers',name="Node",marker=dict(size=10))
        self.fig.add_trace(nodes_trace,row=row,col=col)


def create_sample_dataset():
    adjacency_matrix = np.array(
        [
            [0,1,0,1,0],
            [1,0,1,0,0],
            [0,1,0,0,1],
            [1,0,0,0,1],
            [0,0,1,1,0],
        ]
    )
    return adjacency_matrix 

def main():
    adjacency_matrix = create_sample_dataset()

    graph_autoencoder = GraphAutoencoder(encoding_dim=2)
    graph_autoencoder.build_autoencoder(num_nodes=adjacency_matrix.shape[0])
    graph_autoencoder.compile_autoencoder()

    graph_autoencoder.train_autoencoder(adjacency_matrix=adjacency_matrix,epochs=100,batch_size=1)

    encoded_data = graph_autoencoder.encode(adjacency_matrix=adjacency_matrix)
    decoded_data = graph_autoencoder.decode(encoded_data)

    graph_visualizer = GraphVisualizer()
    graph_visualizer.visualizer_graph(adjacency_matrix,encoded_data,decoded_data)




if __name__ == "__main__":
    main()





    