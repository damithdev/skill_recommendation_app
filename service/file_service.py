import os
import pdfplumber


class FileService:
    @staticmethod
    def extract_text(file_name, file_object):
        """
        Extracts text from a given file object. Supports various file formats.

        :param file_name: The name of the file
        :param file_object: File object to be processed.
        :return: Extracted text as a string, or None if extraction fails.
        """
        try:
            # Extract file extension
            file_extension = os.path.splitext(file_name)[1].lower()

            # Process based on file extension
            if file_extension in ['.pdf']:
                return FileService._extract_text_from_file(file_extension, file_object)
            else:
                print(f"Unsupported file type: {file_extension}")
                return None
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None

    @staticmethod
    def _extract_text_from_file(file_extension, file_object):
        if file_extension == '.pdf':
            return FileService._extract_text_from_pdf(file_object)

    @staticmethod
    def _extract_text_from_pdf(file_object):
        with pdfplumber.open(file_object) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text