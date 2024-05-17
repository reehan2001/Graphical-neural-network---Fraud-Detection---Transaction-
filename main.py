import logging
from Connection import Neo4jConnector
from DataProcessor import DataProcessor
from ModelBuilder import ModelBuilder
from ModelEvaluator import ModelEvaluator
from BubbleGraph import BubbleGraph

NEO4J_URI = "neo4j://localhost:7687" 
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Parsengg@4508"
HIGH_AMOUNT_THRESHOLD = 1000000

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnomalyFinder:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def find_anomalies(self):
        self.model.eval()
        _, pred = self.model(self.data).max(dim=1)
        anomalies = pred[self.data.test_mask].nonzero().tolist()
        return anomalies

def main():
    logging.info("Starting the anomaly detection process.")
    
    connect = Neo4jConnector(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    try:
        connect.connect()
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        return

    if connect.driver is None:
        logging.error("Neo4j driver is None. Exiting.")
        return
    
    try:
        records = connect.load_transaction(HIGH_AMOUNT_THRESHOLD)
    except Exception as e:
        logging.error(f"Error loading transactions: {e}")
        return
    
    fraud_transaction = [(record["id"], record['amount']) for record in records if record.get("fraud")]
    high_amount_transaction = [(record["id"], record['amount']) for record in records if record["amount"] > HIGH_AMOUNT_THRESHOLD]

    connect.close()

    if not fraud_transaction and not high_amount_transaction:
        logging.info("No anomalies found in the dataset.")
        return
    
    data_processor = DataProcessor(fraud_transactions=fraud_transaction, high_amount_transactions=high_amount_transaction, records=records)
    data = data_processor.process_data()

    model_builder = ModelBuilder(data)
    model = model_builder.build_model()

    model_evaluator = ModelEvaluator(model, data)
    model_evaluator.evaluate()

    anomaly_finder = AnomalyFinder(model, data)
    anomalies = anomaly_finder.find_anomalies()

    bubble_graph = BubbleGraph(fraud_transaction, high_amount_transaction)
    bubble_graph.create_graph(anomalies)

    logging.info("Anomaly detection process completed.")

if __name__ == "__main__":
    main()
