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
    
        self.prev_xy = {()}
        self.first_goal = False
        self.reached_first = 0
        self.stuck_counter = 0
        self.num_battle = 0
        self.before_prior_count = 0
        self.location = {"OAKS_LAB,"}
        self.prev_location = ""
        self.location_counter = 0 #Num of times it stays within the same location
        self.inBattle = False
        self.prev_enemy_hp = 0
        self.button_chosed  = 0
    
    def _run_action_on_emulator(self, action_array: np.ndarray) -> None:
        action = action_array[0]
        action = min(action, 0.99)

        # Continuous Action is a float between 0 - 1 from Value based methods
        # We need to convert this to an action that the emulator can understand
        bins = np.linspace(0, 1, len(self.valid_actions) + 1)
        button = np.digitize(action, bins) - 1

        self.button_chosed = button

        # Push the button for a few frames
        self.pyboy.send_input(self.valid_actions[button])

        for _ in range(self.act_freq):
            self.pyboy.tick()

        # Release the button
        self.pyboy.send_input(self.release_button[button])

    def _movement_reward(self, new_state: dict[str, any]) -> int:
        reward = 0
        if self.inBattle:
            return 0
        else:
            # counter_x = 0
            # counter_y = 0
            if ((new_state["location"]["x"], new_state["location"]["y"]) != (self.prior_game_stats["location"]["x"], self.prior_game_stats["location"]["y"])):
                self.stuck_counter = 0
                if (len(self.prev_xy)) == 1:
                    self.prev_xy.add((self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]))
                elif (new_state["location"]["x"], new_state["location"]["y"]) in self.prev_xy:
                    reward += -5
                else:
                    self.prev_xy.add((self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["y"]))
                    reward += 1
                
                # for i in self.prev_x:
                #     if i == new_state["location"]["x"]:
                #         counter_x += 1
                # for j in self.prev_y:
                #     if j == new_state["location"]["y"]:
                #         counter_y += 1

                # if (counter_x > 2) or (counter_y > 2):
                #     reward -= 1#4#counter_y + counter_x
                
                if new_state["location"]["map"] == "PALLET_TOWN,":
                    if (new_state["location"]["x"], new_state["location"]["y"]) == (11,0):
                        #if self.first_goal == False: #and self.reached_first < 10:
                            #self.reached_first += 1
                        reward += 5#5
                        # else:
                        #     reward += 2
                    elif new_state["location"]["x"] in [10, 12] and new_state["location"]["y"] in [0, 1]:
                        #if self.first_goal == False: #and self.reached_first < 10:
                            #self.reached_first += 1
                        reward +=2 #2
                        # else:
                        #     reward += 1
                elif new_state["location"]["map"] == "ROUTE_1,":
                    reward +=8#10
                
                self.before_prior_count += 1
                reward += 1 
            else:
                self.stuck_counter += 1
                if self.stuck_counter > 5:
                    reward -= min(5, self.stuck_counter) #slowly increases from 5 to 20
                          
            return reward
    
    def _location_reward(self, new_state: dict[str, any]) -> float :
        reward = 0
        if self.inBattle:
            return 0
        else:
            #new location, haven't visited before
            if new_state["location"]["map"] not in self.location and (new_state["location"]["map"] != self.prior_game_stats["location"]["map"]) :
                self.location_counter = 0
                self.location.add(new_state["location"]["map"])
                reward += 30 #100
            #New location, but visited before
            elif (new_state["location"]["map"] != self.prior_game_stats["location"]["map"]) and new_state["location"]["map"] in self.location:
                self.location_counter = 0
                #To prevent abuse of resetting the location counter
                if self.prev_location == new_state["location"]["map"]:
                    reward -= 2
                self.prev_location = self.prior_game_stats["location"]["map"]
                if new_state["location"]["map"] == "ROUTE_1,":
                    reward += 5#self._grass_reward(new_state)*6 #the grass reward always return 1
                else:
                    reward += 2#1
            #SAme location and has visited before
            elif (new_state["location"]["map"] == self.prior_game_stats["location"]["map"]) and new_state["location"]["map"] in self.location:
                reward -= 1#4
                self.location_counter +=1 #for number of times stuck in the same map
                #if stuck in route 1, reward enough that the taking reward part isn't that affected
                if new_state["location"]["map"] == "ROUTE_1,":
                    reward += 4#self._grass_reward(new_state)*6 #the grass reward always return 1
                if self.location_counter > 10:
                    reward -= 1 + min(20, self.location_counter)#4
        
            return reward
    
    
    def _in_battle_reward(self, new_state: dict[str, any]) -> float :
        reward = 0
        if self._read_m(0xD057) != 0:
            self.inBattle = True
            #self.num_battle += 1
            enemy_curr_hp = self._read_hp(0xCFE6) #self._read_m(0xCFE6) | self._read_m(0xCFE7)
            enemy_max_hp = self._read_hp(0xCFF4)  #self._read_m(0xCFF4) | self._read_m(0xCFF5)
            self.prev_enemy_hp = enemy_max_hp
            
            #for very first battle
            # if self.button_chosed == 4 and self.num_battle == 0:
            #     reward += 20#75
            # elif self.button_chosed == 4 and (self.num_battle > 0):
            #     reward += 10#50
            # else:
            #     reward -= 1#1
            
            if enemy_curr_hp > self.prev_enemy_hp and enemy_curr_hp > 0:
                self.prev_enemy_hp = enemy_curr_hp
                reward += 10 #75  # Reward for dealing damage
            elif enemy_curr_hp == 0:
                self.inBattle = False
                self.num_battle += 1
                reward += 30#100  # Defeated enemy
            elif enemy_curr_hp <= self.prev_enemy_hp:
                reward += -1#-5  # No change in enemy HP (reduced penalty)

            return reward
        else:
            self.inBattle = False
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
        self.prev_xy.add((self.prior_game_stats["location"]["x"],self.prior_game_stats["location"]["x"]))
        seen_reward = self._seen_reward(new_state)
        
        battle_reward = self._in_battle_reward(new_state)
        xp_reward = self._xp_reward(new_state)


        caught_reward = self._caught_reward(new_state)

        if new_state["caught_pokemon"] == 1:
            caught_reward =caught_reward * 30
        elif new_state["caught_pokemon"] == 2:
            caught_reward= caught_reward * 35
        elif new_state["caught_pokemon"] == 3:
            caught_reward= caught_reward * 20
        elif new_state["caught_pokemon"] >= 4:
            caught_reward= caught_reward * 10


        total_reward = (self._movement_reward(new_state) * 0.4
                        + self._location_reward(new_state) * 0.7
                        + seen_reward *0.5
                        + caught_reward *0.8
                        + battle_reward * 1.0
                        + xp_reward*0.8)

        return total_reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        if self.steps >= 1000:
            self.prev_xy = {()}
            self.first_goal = False
            self.reached_first = 0
            self.stuck_counter = 0
            self.num_battle = 0
            self.before_prior_count = 0
            self.location = {"OAKS_LAB,"}
            self.location_counter = 0 #Num of times it stays within the same location
            self.inBattle = False
            self.prev_enemy_hp = 0
            self.button_chosed  = 0
            return 1
        else:
            return 0
        
        # Maybe if we run out of pokeballs...? or a max step count
        #return self.steps >= 1000
