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
        self.prev_loc = {""}
        self.in_battle = True
        self.different_loc = False
        self.complete_stuck = 0


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

        if self.in_battle:
            return 0
        else:
            if ((new_state["location"]["y"]) < (self.prior_game_stats["location"]["y"])) and (new_state["location"]["map"] != "OAKS_LAB,"):
                #may add additional rewards for reaching hear y = 0
                if self.y_reward == 2:
                    self.y_reward = 0 #attempt to make it to top by 100 steps
                if 2 <((new_state["location"]["y"]) - 11) < 5:
                    reward += 3
                elif ((new_state["location"]["y"]) - 11) <= 2:
                    reward += 4
                reward += 1 + min(self.y_reward, 1.5)
                self.y_reward += 0.1

            if [(new_state["location"]["x"], new_state["location"]["y"]) != (self.prior_game_stats["location"]["x"], self.prior_game_stats["location"]["y"])]:
                if self._grass_reward(new_state) :
                    reward += 2
                reward += 1
            else:
                self.complete_stuck += 1
                reward = 0
                
            
            return reward                  
    
    def _location_reward(self, new_state: dict[str, any]) -> float :
        reward = 0
        if self.in_battle:
            return 0
        else:
                
            if new_state["location"]["map"] not in self.prev_loc:
                self.prev_loc.add(new_state["location"]["map"])
                self.different_loc = False
                reward += 400
            elif new_state["location"]["map"] != self.prior_game_stats["location"]["map"] and self.different_loc == True:
                self.different_loc = False #After going into new location that is already visited, cannot come and get this reward until 200 steps has passed
                reward += 100
            elif new_state["location"]["map"] == self.prior_game_stats["location"]["map"]:
                reward += 50 - min(self.same_loc, 50)            #may need to change this from += to -= and - to +
                self.same_loc += 1                  
                #after being stuck for 200 ticks
                if self.same_loc == 2:
                    self.different_loc = True
                    reward -= 1
                    self.same_loc = 0
            else:
                reward =0
            return reward
    
    
    def _in_battle_reward(self) -> float :
        reward = 0
        if self._read_m(0xD057) != 0:
            enemy_curr_hp = self._read_hp(0xCFE6) #self._read_m(0xCFE6) | self._read_m(0xCFE7)
            enemy_max_hp = self._read_hp(0xCFF4)  #self._read_m(0xCFF4) | self._read_m(0xCFF5)
            turn_num = self._read_m(0xCCD5) #turn number starts from 0

            if ((enemy_max_hp/2) < enemy_curr_hp < (enemy_max_hp)) and turn_num != self.prev_turn:
                reward += 400
                self.prev_turn = turn_num
            elif (0 < enemy_curr_hp < (enemy_max_hp/2)) and turn_num != self.prev_turn:
                reward += 200
                self.prev_turn = turn_num
            elif (enemy_curr_hp == 0) :
                reward += 500
                self.in_battle = False
            elif turn_num == self.prev_turn:
                reward -= 0 + min(self.turn_unreward, 1)
                #when it is continously stuck on the same turn
                if turn_num < self.prev_turn:
                    self.turn_unreward += 0.01
                if self.turn_unreward == 2:
                    self.turn_unreward = 0
                self.prev_turn = turn_num
            self.in_battle = True
        else:
            self.in_battle = False
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

        caught_reward = self._caught_reward(new_state)

        total_reward = (movement_reward * 0.5
                        + location_reward * 0.8
                        + seen_reward *0.4
                        + caught_reward *0.75
                        + battle_reward * 0.9
                        + xp_reward*0.5)

        return total_reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        if self.steps >= 1000:
            self.same_loc = 0
            self.turn_unreward = 0 
            self.y_reward = 0
            self.prev_turn = -1
            self.prev_loc = {""}
            self.in_battle = True
            self.different_loc = False
            self.complete_stuck = 0
            return 1
        elif self.complete_stuck >= 100:
            self.same_loc = 0
            self.turn_unreward = 0 
            self.y_reward = 0
            self.prev_turn = -1
            self.prev_loc = {""}
            self.in_battle = True
            self.different_loc = False
            self.complete_stuck = 0
            return 1
        else:
            return 0
