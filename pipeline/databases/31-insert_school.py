#!/usr/bin/env python3
"""Insert a new document in a MongoDB collection based on kwargs"""


def insert_school(mongo_collection, **kwargs):
    """Inserts a new document into mongo_collection using kwargs
    Args:
        mongo_collection: pymongo collection object
        **kwargs: key-value pairs to insert as the new document
    Returns:
        The _id of the newly inserted document
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
