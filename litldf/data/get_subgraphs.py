import numpy as np
from litldf.utils.cache import cached_with_io


@cached_with_io
def get_subgraph_nodes(meters_to_bldgs,
                       bldgs_to_meters,
                       meters_queue,
                       bldgs_queue,
                       meters_history,
                       bldgs_history):
    finished = False

    while not finished:

        bldg = ''
        meter = ''

        # initialize to single meter
        if len(meters_queue) == 0 and \
                len(bldgs_queue) == 0 and \
                len(meters_history) == 0 and \
                len(bldgs_history) == 0:
            bldg = ''
            meter = list(meters_to_bldgs.keys())[0]
        # if there's at least one meter in queue, let's select a meter
        elif len(meters_queue) > 0:
            bldg = ''
            meter = meters_queue[0]
            meters_queue.remove(meter)
        # if there's at least one building in queue, let's select a building
        elif len(bldgs_queue) > 0:
            meter = ''
            bldg = bldgs_queue[0]
            bldgs_queue.remove(bldg)

        # if meter is selected, add connected bldgs to queue
        if meter != '':
            # add connected buildings to queue
            new_bldgs = meters_to_bldgs[meter]
            bldgs_queue.extend(new_bldgs)
            bldgs_queue = np.unique(bldgs_queue).tolist()

            # remove meters in history from meters queue
            bldgs_queue = np.setdiff1d(bldgs_queue, bldgs_history).tolist()

            # save meter to history
            meters_history.append(meter)

        # if bldg is selected, add connected meters to queue
        if bldg != '':
            # add connected meters to queue
            new_meters = bldgs_to_meters[bldg]
            meters_queue.extend(new_meters)
            meters_queue = np.unique(meters_queue).tolist()

            # remove meters in history from meters queue
            meters_queue = np.setdiff1d(meters_queue, meters_history).tolist()

            # save bldg to history
            bldgs_history.append(bldg)

        # check if finished
        if len(meters_queue) == 0 and \
                len(bldgs_queue) == 0 and \
                len(meters_history) > 0 and \
                len(bldgs_history) > 0:
            finished = True

    return meters_history, bldgs_history


def main(args):
    meters_to_bldgs = {'a': ['1', '2', '3'],
                       'b': ['3', '4'],
                       'c': ['1', '2', '5', '6', '7'],
                       'd': ['6', '7', '8'],
                       'e': ['9']}

    bldgs_to_meters = {'1': ['a', 'c'],
                       '2': ['a', 'c'],
                       '3': ['a', 'b'],
                       '4': ['b'],
                       '5': ['c'],
                       '6': ['c', 'd'],
                       '7': ['c', 'd'],
                       '8': ['d'],
                       '9': ['e']}

    meters_history, bldgs_history = \
        get_subgraph_nodes(meters_to_bldgs,
                           bldgs_to_meters,
                           [],
                           [],
                           [],
                           [])


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
