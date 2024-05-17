from neo4j import GraphDatabase
import logging

class Neo4jConnector:
    def __init__(self,url,user,password):
        self.url = url
        self.user = user
        self.password = password
        self.driver = None
    
    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.url,auth={self.user,self.password})
            print("connected sucessfully")
        except Exception as e:
            logging.error("Failed to connect" , e)
            self.driver = None
    
    def close(self):
        if self.driver:
            self.driver.close()

    def execute_cypher_query(self,query):
        with self.driver.session() as session:
            result = session.run(query)
            return list(result)
        
    def load_transaction(self,high_amount_threshold):
        query = f'''
        MATCH (t:Trasaction)
        WHERE t.fraud = true OR t.amount > {high_amount_threshold}
        RETURN t.id AS id, t.fraud,t.amount AS amount
        '''
        return self.execute_cypher_query(query)
