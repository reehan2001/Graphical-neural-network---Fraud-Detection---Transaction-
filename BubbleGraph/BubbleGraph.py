import plotly.graph_objects as go

class BubbleGraph:
    def __init__(self,fraud_transactions,high_amount_transactios):
        self.fraud_transactions = fraud_transactions
        self.high_amount_transactions = high_amount_transactios

    def create_graph(self,anomalies):
        fraud_transaction_x = [transaction_id for transaction_id , _ in self.fraud_transactions]
        fraud_transaction_y = [amount for _, amount in self.fraud_transactions]
        high_amount_tranctions_x = [transaction_id for transaction_id , _ in self.high_amount_transactions]
        high_amount_tranctions_y = [amount for _, amount in self.high_amount_transactions]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fraud_transaction_x,
                y=fraud_transaction_y,
                mode = "markers",
                marker = dict(color='red'),
                name = 'proven Fraud'
            )
        )

        fig.add_trace(
            go.scatter(
                x = high_amount_tranctions_x,
                y = high_amount_tranctions_y,
                mode = "markers",
                markers = dict(
                    color = high_amount_tranctions_y,
                    colorscale = "Rainbow",
                    colorbar= dict(
                        title = 'Transaction Amount',
                        len = 0.8
                    )

                ),
                name = "Maybe Fraud due to high amount"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=anomalies,
                y = [0]*len(anomalies),
                mode = "markers",
                marker=dict(color='black'),
                name= "Predicted Anomalies"
            )
        )

        fig.show()

    

