import requests
import json
from math import ceil
import os
import time
import pandas as pd
from VirkResult import VirkResult

class VirkConnection():
    """ Class for connecting to the CVR API of Virk.dk. Utilizes the scroll api to fetch all results from a given query. """

    def __init__(self, credentials, query, endpoint, MAX_ITERATIONS=50):
        """
        :param credentials: dict, containing the user and password for the API
        :param query: dict, containing the query to be executed 
        :param endpoint: str, the endpoint to be queried (either "cvr-permanent/virksomhed", "cvr-permanent/produktionsenhed", "registreringstekster/registreringstekst" or "offentliggoerelser")
        """

        self.credentials: dict = credentials
        self.query: dict = query
        self.endpoint: str = endpoint
        self.MAX_ITERATIONS: int = MAX_ITERATIONS

        self.scroll_id: str = None
        self.total_hits: int = None
        self.chunk_size: int = None
        self.active_connection: bool = False
        self.data: list[VirkResult] = []
        self.parsed_data: list = None
        assert endpoint in ["cvr-permanent/virksomhed", "cvr-permanent/produktionsenhed", "registreringstekster/registreringstekst", "offentliggoerelser"], "Invalid endpoint"


    def parse_results(self):
        """Parses and saves the data from the fetched response
        :param data: list of dictionaries, containing the data to be parsed
        """

        if self.endpoint == 'offentliggoerelser':
            self.parsed_data = []
            for hit_list in self.data:
                self.parsed_data += hit_list.parse(endpoint=self.endpoint)
 
        elif self.endpoint == 'cvr-permanent/virksomhed':
            self.parsed_data = [[],[],[],[],[]]
            for hit_list in self.data:
                result = hit_list.parse(endpoint=self.endpoint)
                self.parsed_data[0] += result[0] # registrations 
                self.parsed_data[1] += result[1] # employee counts 
                self.parsed_data[2] += result[2] # production units
                self.parsed_data[3] += result[3] # company info 
                self.parsed_data[4] += result[4] # participants

                del result

                
        elif self.endpoint == 'registreringstekster/registreringstekst':
            self.parsed_data = []
            for hit_list in self.data:
                self.parsed_data += hit_list.parse(endpoint=self.endpoint)
        

        elif self.endpoint == 'cvr-permanent/produktionsenhed':
            # return pandas dataframe with columns [pNummer, Municipality, Industry]
            self.parsed_data = []
            for hit_list in self.data:
                self.parsed_data += hit_list.parse(endpoint=self.endpoint)
                
        else:
            raise NotImplementedError('Parsing for this endpoint is not implemented yet.')

        
        
    def start_scroll(self):
        self.time_start = time.time()
        url = f'http://distribution.virk.dk/{self.endpoint}/_search?scroll=1m'
        response = requests.post(url, auth=(self.credentials['user'], self.credentials['password']), 
                            json=self.query)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(response.text)
            raise Exception('Error starting scroll')

        # fetch data
        output = response.json()
        data = output['hits']['hits']
        self.data.append(VirkResult(data))


        # update the class variables
        self.iteration = 0
        self.active_connection = True
        self.scroll_id = output['_scroll_id']
        self.total_hits = output['hits']['total']
        print(f"Total hits: {self.total_hits}")
        self.chunk_size = len(data)



    def update_scroll(self):
        """ 
        Fetches next batch of data from the scroll. Terminates if no more data is available.
        """
        print(f"Iteration {self.iteration+1} of {ceil(self.total_hits / self.chunk_size)}")
        url = f'http://distribution.virk.dk/_search/scroll'
        response = requests.get(url, auth=(self.credentials['user'], self.credentials['password']), 
                                json={"scroll":"1m", "scroll_id": self.scroll_id})
        
        # Check if the request was successful
        if response.status_code != 200:
            self.delete_scroll(self.scroll_id)
            print(response.text)
            raise Exception('Error updating scroll')
        
        # fetch the response
        output = response.json()
        data = output['hits']['hits']
        self.data.append(VirkResult(data))

        # update class variables
        self.scroll_id = output['_scroll_id']
        self.iteration += 1
        
        # Check if all data has been fetched
        if len(data) == 0:
            print('SUCCESS: All data from the query has been fetched')
            self.delete_scroll()
            return


            
    def delete_scroll(self):
        url = f'http://distribution.virk.dk/_search/scroll'
        response = requests.delete(url, auth=(self.credentials['user'], self.credentials['password']), 
                            json={"scroll_id": self.scroll_id})
        if response.status_code == 200:
            print('-> Scroll deleted')
            self.scroll_id = None
        else:
            print(f'Error ocurred when trying to delete scroll {self.scroll_id}')


    def execute_query(self):
        """ 
        Fetches all results from a given query using the 'scroll' API of Elasticsearch, which continuously sends requests to the server.
        Note: The time window to complete the scroll is 1 minute. Filter queries accordingly.
        :param MAX_ITERATIONS: int, maximum number of iterations allowed before breaking the request loop
        
        """

        # start the scroll
        self.start_scroll()
        total_iterations = ceil(self.total_hits / self.chunk_size)

        # inform about the query
        print(f"Fetching {self.total_hits} documents in chunks of {self.chunk_size}")
        print(f"{total_iterations} iterations required.")
        
        # safety measure: Break if more than a given iterations are required
        if total_iterations > self.MAX_ITERATIONS:
            self.delete_scroll()
            raise Exception('Too many iterations required. Filter the query to return less results.')

        # update while connection is active   
        try:
            for i in range(total_iterations):
                if self.scroll_id != None:
                    self.update_scroll()
                else:
                    break
        finally:
            # error handling: delete scroll if an error occurs
            if self.scroll_id != None:
                self.delete_scroll()
        

        print(f"Fetched {self.total_hits} documents in {time.time() - self.time_start} seconds.")
        if self.scroll_id != None:
            print("Closing connection...")
            self.delete_scroll()
        else:
            print("Connection already closed.")