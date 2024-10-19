import grpc

TIMEOUT_SEC = 15
def grpc_server_on(channel) -> bool:
    try:
        grpc.channel_ready_future(channel).result(timeout=TIMEOUT_SEC)
        return True
    except grpc.FutureTimeoutError:
        return False

# Create a gRPC channel before calling the function
channel = grpc.insecure_channel('localhost:50051')  # Update the address as needed
print(grpc_server_on(channel))
