# Things i should be using while learning asyncIO
#     1. Typing
#     2. Logging
#     3. Asyncio
#     4. random
#     5. time (maybe)
#     6. OOP
from typing import Callable
import asyncio
import logging
from abc import ABC
import time
from functools import wraps

class Logger:
    def __init__(self) -> None:
        logger = logging.basicConfig(filemode='w', level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(msg)s")
        return None

class OrderWork:
    def __init__(self, name:str) -> None:
        self.name = name
        self.logging = logging.getLogger(__name__)

    @staticmethod
    def bind_with_time(f:Callable)->Callable:
        wraps(f)
        def func(*args, **kwargs)->float:
            start = time.perf_counter()
            f(*args, **kwargs)
            end = time.perf_counter()
            total_time_spent = end - start
            return total_time_spent
        return func
    
    @bind_with_time
    def start_work(self)->None:
        self.logging.info(f"Starting to boil Milk for {self.name}")
        time.sleep(1)
        self.logging.info(f"Putting some chai patti, sugar and other topings for {self.name}")
        time.sleep(1)
        self.logging.info(f"Serving the tea to {self.name}")
        time.sleep(1)
        self.logging.info(f"{self.name} liked the tea")

    # @bind_with_time
    async def astart_work(self)->None:
        self.logging.info(f"Starting to boil Milk for {self.name}")
        await asyncio.sleep(1)
        self.logging.info(f"Putting some chai patti, sugar and other topings for {self.name}")
        await asyncio.sleep(1)
        self.logging.info(f"Serving the tea to {self.name}")
        await asyncio.sleep(1)
        self.logging.info(f"{self.name} liked the tea")

class Main:
    @staticmethod
    async def main() -> None:
        o1 = OrderWork("Ansh")
        o2 = OrderWork("Ajay")
        start = time.perf_counter()
        await asyncio.gather(o1.astart_work(), o2.astart_work())
        end = time.perf_counter()
        total_time_spent = end - start
        print(f"Total taken, is {total_time_spent}")

if __name__ == "__main__":
    logger = Logger()
    # o1 = OrderWork("Ansh")
    # o2 = OrderWork("Ajay")
    # time_spent_doing_o1 = o1.start_work()
    # time_spent_doing_o2 = o2.start_work()
    # print(f"Total taken, is {time_spent_doing_o1 + time_spent_doing_o2}")
    asyncio.run(Main.main())
