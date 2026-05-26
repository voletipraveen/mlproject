from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    This function will take input from the req.txt file as sting and returns the list of requriments 
    '''

    requirements =[]
    HYPEN_E_DOT="-e."
    with open("requirements.txt") as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] #Removes the \n from the above list

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name="mlproject",
    version='0.0.1',
    author="Praveen",
    author_email="voletipraveen370@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirments.txt")
)