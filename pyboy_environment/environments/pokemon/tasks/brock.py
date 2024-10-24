from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            #WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )
    
        self.same_loc = 0
        self.turn_unreward = 0 
        self.y_reward = 0
        self.prev_turn = -1
        self.prev_loc = set()
        self.prev_spot = set()
        self.no_move = 0

    def reset(self) -> np.ndarray:
        self.steps = 0

        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)

        self.prior_game_stats = self._generate_game_stats()

        self.same_loc = 0
        self.turn_unreward = 0 
        self.y_reward = 0
        self.prev_turn = -1
        self.prev_loc = set()
        self.prev_spot = set()
        self.no_move = 0


        return self._get_state()

    def _run_action_on_emulator(self, action_array: np.ndarray) -> None:
        action = action_array[0]
        action = min(action, 0.99)

        # Continuous Action is a float between 0 - 1 from Value based methods
        # We need to convert this to an action that the emulator can understand
        bins = np.linspace(0, 1, len(self.valid_actions) + 1)
        button = np.digitize(action, bins) - 1

        # Push the button for a few frames
        self.pyboy.send_input(self.valid_actions[button])

        for _ in range(self.act_freq):
            self.pyboy.tick()

        # Release the button
        self.pyboy.send_input(self.release_button[button])

    def _movement_reward(self, new_state: dict[str, any]) -> int:
        reward = 0

        if (new_state["location"]["x"], new_state["location"]["y"]) != (self.prior_game_stats["location"]["x"], self.prior_game_stats["location"]["y"]):
            #new coordinate unlocked
            if (new_state["location"]["x"], new_state["location"]["y"]) not in self.prev_spot:
                self.prev_spot.add((new_state["location"]["x"], new_state["location"]["y"]))
                if new_state["location"]["map"] == "PALLET_TOWN,":
                    reward += 2
                elif new_state["location"]["map"] == "ROUTE_1,":
                    reward += 3
                elif new_state["location"]["map"] == "VIRIDEAN_CITY,":
                    reward += 4
                elif new_state["location"]["map"] == "REDS_HOUSE_1F" or new_state["location"]["map"] == "REDS_HOUSE_2F" or new_state["location"]["map"] == "BLUES_HOUSE":
                    reward -= 1
            else:
                if new_state["location"]["map"] == "OAKS_LAB,":
                    reward -= 1

            if self._grass_reward(new_state) :
                reward += 1
            reward += 1
            #print("Has moved reward")

        if (new_state["location"] == self.prior_game_stats["location"] and self._read_m(0xD057) == 0):
            self.no_move += 1
            reward -= 1
        if new_state["location"]["map"] == "PALLET_TOWN,":
            if (new_state["location"]["y"]) < (self.prior_game_stats["location"]["y"]): #to attempt to go up
                reward += 4 #<-Change
                #print("Heading entrance of ROUTE_1")
            elif (new_state["location"]["y"]) > (self.prior_game_stats["location"]["y"]):
                reward -= 2
                #print("Away from entrance of ROUTE_1")
            elif 0 <= (new_state["location"]["y"]) < 3 and  (9 <new_state["location"]["y"] < 13):
                reward += 4.5
            #print("Near entrance of ROUTE_1")

        return reward                  
    
    def _location_reward(self, new_state: dict[str, any]) -> float :
        reward = 0

        #New area 
        if new_state["location"]["map"] not in self.prev_loc:
            self.prev_loc.add(new_state["location"]["map"])
            reward = 50
            if new_state["location"]["map"] == "PALLET_TOWN,":
                reward += 2
            elif new_state["location"]["map"] == "ROUTE_1,":
                reward += 3
            elif new_state["location"]["map"] == "VIRIDEAN_CITY,":
                reward += 4
            elif new_state["location"]["map"] == "REDS_HOUSE_1F" or new_state["location"]["map"] == "REDS_HOUSE_2F" or new_state["location"]["map"] == "BLUES_HOUSE":
                reward -= 1
           #print("Newly visited location")
        # elif new_state["location"]["map"] in self.prev_loc:
        #     reward = 10
            #print("Different location from previously")
        
        # if new_state["location"]["map"] == "OAKS_LAB,":
        #     reward -= 5

        if new_state["location"]["map"] == "OAKS_LAB," or new_state["location"]["map"] == "REDS_HOUSE_1F" or new_state["location"]["map"] == "REDS_HOUSE_2F" or new_state["location"]["map"] == "BLUES_HOUSE":
            reward -= 1

        return reward
    
    def _in_battle_reward(self) -> float :
        reward = 0
        if self._read_m(0xD057) != 0:
            enemy_curr_hp = self._read_hp(0xCFE6) #self._read_m(0xCFE6) | self._read_m(0xCFE7)
            enemy_max_hp = self._read_hp(0xCFF4)  #self._read_m(0xCFF4) | self._read_m(0xCFF5)
            turn_num = self._read_m(0xCCD5) #turn number starts from 0

            if ((enemy_max_hp/2) < enemy_curr_hp < (enemy_max_hp)) and turn_num > 0:
                reward = 4
                #print("Has dealt damage")
                self.prev_turn = turn_num
            elif (0 < enemy_curr_hp < (enemy_max_hp/2)) and turn_num > 0:
                reward = 5
                #print("Damaged below half")
                self.prev_turn = turn_num
            elif turn_num == self.prev_turn:
                #print("stuck in same turn")
                reward = -1
                self.prev_turn = turn_num
            else:

                reward = 0
        else:
            #print("Not in batte")
            reward = 0

        return reward


    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()

        x = np.array([game_stats["location"]["x"]])
        y = np.array([game_stats["location"]["y"]])
        location = np.array([game_stats["location"]["map_id"]])

        hp_list = game_stats["hp"]["current"]
        curr_hp = 0
        for i in hp_list:
            curr_hp = curr_hp + i
        curr_hp = np.array([curr_hp])
        enemy_curr_hp = np.array([self._read_hp(0xCFE6)])
        num_caught = np.array([game_stats["caught_pokemon"]])

        state = np.concatenate([x, y, 
                                location,
                                curr_hp,
                                enemy_curr_hp,
                                num_caught]).flatten()
   
        return state#, np.array(self.pyboy.game_area())]#state  #[game_stats["location"]["map_id"], game_stats["seen_pokemon"], game_stats["caught_pokemon"]]

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        
        location_reward = self._location_reward(new_state)
        movement_reward = self._movement_reward(new_state)
        seen_reward = self._seen_reward(new_state)
        battle_reward = self._in_battle_reward()
        xp_reward = self._xp_reward(new_state)
        #print(f"Location: {location_reward}, Movement: {movement_reward}, Seen: {seen_reward}, battle: {battle_reward}, xp: {xp_reward}")

        caught_reward = self._caught_reward(new_state)

        #Try multiplying all by 10?
        total_reward = (movement_reward #* 0.4
                        + location_reward #* 0.8
                        + seen_reward #*0.4
                        + caught_reward #*0.75
                        + battle_reward #* 0.9  
                        + xp_reward )#*0.5)

        return total_reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        if self.steps >= 1000:
            return 1
        # elif self.no_move >= 500:
        #     return 1
        else:
            return 0
        
        # Maybe if we run out of pokeballs...? or a max step count
        #return self.steps >= 1000