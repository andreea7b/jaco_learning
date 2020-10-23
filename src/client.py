import cPickle as pickle
import socket

class Client():
    def __init__(self, port):
        self.port = port

    def replan(self, start, goal, goal_pose, weight_idx, T, timestep, start_time=0.0,
			   seed=None, return_plan=False, return_both=False):
        query = [
            start,
            goal,
            goal_pose,
            weight_idx,
            T,
            timestep,
            start_time,
        	seed,
            return_plan,
            return_both
        ]
        query = [0, query]
        return self.query_server(query)[0]

    def replan_and_get_cost(self, start, goal, goal_pose, weight_idx, T, timestep, start_time=0.0,
                            seed=None, return_plan=False, return_both=False, add_pose_penalty=False,
                            zero_learned_cost=False):
        query = [
            start,
            goal,
            goal_pose,
            weight_idx,
            T,
            timestep,
            start_time,
        	seed,
            return_plan,
            return_both,
            add_pose_penalty,
            zero_learned_cost
        ]
        query = [2, query]
        return self.query_server(query)

    def get_cost(self, waypts, weight_idx, add_pose_penalty=False, zero_learned_cost=False):
        query = [
            waypts,
            weight_idx,
            add_pose_penalty,
            zero_learned_cost
        ]
        query = [1, query]
        return self.query_server(query)[0]

    def query_server(self, query):
        server_address = ('localhost', self.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(server_address)
        try:
            sock.sendall(pickle.dumps(query, protocol=2))
            sock.shutdown(1)
            output_bytes = bytearray()
            while True:
                data = sock.recv(4096)
                if data:
                    output_bytes.extend(data)
                else:
                    break
            output = pickle.loads(str(output_bytes))
        finally:
            sock.close()
        return output
