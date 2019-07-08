def convert_constant(node, params, layers, node_name):
    """
    Convert Constant layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    print("CONSTANT: %s %s" % (type(params["value"]), params["value"]))
    layers[node_name] = params['value']
