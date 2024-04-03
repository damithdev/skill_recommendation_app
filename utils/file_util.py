import datetime
import uuid


def generate_unique_filename(original_filename):
    """
    Generates a unique file name using a UUID and a datetime stamp.

    :param original_filename: The original file name, used to preserve the file extension.
    :return: A unique file name string.
    """
    # Extract the file extension
    file_extension = original_filename.split('.')[-1]

    # Generate a datetime stamp
    datetime_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Generate a random UUID
    unique_id = uuid.uuid4()

    # Construct the new file name
    new_filename = f"{datetime_stamp}_{unique_id}.{file_extension}"

    return new_filename
