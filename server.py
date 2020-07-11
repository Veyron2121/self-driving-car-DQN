import zmq
import json

from Controllers.HandCraftedController import HandCraftedController

context = zmq.Context()
# noinspection PyUnresolvedReferences
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

controller = HandCraftedController(5, 10, 0.15)

while True:
    #  Wait for next request from client
    message = socket.recv()
    # unpack JSON
    data = json.loads(message)
    print("Received request: %s" % message)

    next_action = controller.get_car_policy(data["velocity"], data["angle_from_road"],
                                            data["distance_from_road"], data["image_path"])

    #  Send reply back to client
    message_2 = str(next_action[0]) + ',' + str(next_action[1])
    print(message_2)

    socket.send(message_2.encode('ascii'))
