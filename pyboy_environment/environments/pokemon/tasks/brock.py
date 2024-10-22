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
            WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
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
    
        self.prev_x = [0,0,0,0,0]
        self.prev_y = [0,0,0,0,0]
        self.movement_counter = 0
        self.location = ["OAKS_LAB,"]
        self.location_counter = 0 #Num of times it stays within the same location
        self.inBattle = False
    
    def _movement_reward(self, new_state: dict[str, any]) -> int:
        reward = 0
        if self.inBattle:
            return 0
        else:
            if new_state["location"]["x"] != self.prior_game_stats["location"]["x"] or new_state["location"]["y"] != self.prior_game_stats["location"]["y"]:
                # if (new_state["location"]["y"] == 0) and (new_state["location"]["x"] == 11) and new_state["location"]["map"] == "PALLET_TOWN,":
                #     reward += 20  
                # if ((new_state["location"]["y"] == 1) and (new_state["location"]["x"] == 9) or (new_state["location"]["x"] == 12)) and new_state["location"]["map"] == "PALLET_TOWN,":
                #     reward += 10
                if new_state["location"]["map"] == "PALLET_TOWN,":
                    if (new_state["location"]["x"] == 11) and (new_state["location"]["y"] == 0):
                        reward += 20
                    elif new_state["location"]["x"] in [9, 12] and new_state["location"]["y"] == 1:
                        reward += 10
                reward += 2
            else:
                self.movement_counter += 1
                if self.movement_counter > 5:
                    reward -= min(20, self.movement_counter) #slowly increases from 5 to 20
                
            
            return reward
    
    def _location_reward(self, new_state: dict[str, any]) -> float :
        reward = 0
        if self.inBattle:
            return 0
        else:
            if new_state["location"]["map"] != self.prior_game_stats["location"]["map"]:
                if new_state["location"]["map"] not in self.location:
                    self.location.append(new_state["location"]["map"])
                    reward += 100
                else:
                #still in the same location as
                    self.location_counter += 1
                    reward += 2 + (self._grass_reward(new_state))*2
                    if self.location_counter > 10:
                        self.location_counter = 0
                        reward -= 4
        
            return reward
    
    
    def _in_battle_reward(self, new_state: dict[str, any]) -> float :
        if self._read_m(0xD057) != 0:
            self.inBattle = True
            enemy_curr_hp = self._read_hp(0xCFE6) #self._read_m(0xCFE6) | self._read_m(0xCFE7)
            enemy_max_hp = self._read_hp(0xCFF4)  #self._read_m(0xCFF4) | self._read_m(0xCFF5)
            #print("enemy hp", enemy_curr_hp)
            # if enemy_curr_hp == enemy_max_hp:
            #     self.prev_enemy_hp = enemy_curr_hp
            #     return -10
            # elif enemy_curr_hp == self.prev_enemy_hp:
            #     return -20
            # elif (0< enemy_curr_hp < enemy_max_hp) :
            #     #self.display_print = True
            #     return 50
            # elif (enemy_curr_hp == 0):
            #     self.prev_enemy_hp = 0
            #     self.inBattle = False
            #     return 100
            if enemy_curr_hp < enemy_max_hp and enemy_curr_hp > 0:
                self.prev_enemy_hp = enemy_curr_hp
                return 50  # Reward for dealing damage
            elif enemy_curr_hp == 0:
                self.inBattle = False
                return 100  # Defeated enemy
            elif enemy_curr_hp == self.prev_enemy_hp:
                return -5  # No change in enemy HP (reduced penalty)
            else:
                return -10  # General penalty for entering battles with no action
        else:
            self.inBattle = False
            return 0


    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()
        return [np.array(self.pyboy.game_wrapper._get_screen_background_tilemap())]#, np.array(self.pyboy.game_area())]

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here

        seen_reward = self._seen_reward(new_state)
        
        battle_reward = self._in_battle_reward(new_state)
        xp_reward = self._xp_reward(new_state)


        caught_reward = self._caught_reward(new_state)

        if new_state["caught_pokemon"] == 1:
            caught_reward =caught_reward * 50
        elif new_state["caught_pokemon"] == 2:
            caught_reward= caught_reward * 40
        elif new_state["caught_pokemon"] == 3:
            caught_reward= caught_reward * 20
        elif new_state["caught_pokemon"] >= 4:
            caught_reward= caught_reward * 10


        total_reward = (self._movement_reward(new_state) * 0.5
                        + self._location_reward(new_state) * 0.7
                        + seen_reward *0.4
                        + caught_reward *0.8
                        + battle_reward * 1.0
                        + xp_reward*0.8)

        return total_reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
