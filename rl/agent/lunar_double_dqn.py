from agent.double_dqn import DoubleDQN


class LunarDoubleDQN(DoubleDQN):

    def update_e(self, sys_vars, replay_memory):
        '''strategy to update epsilon'''
        super(LunarDoubleDQN, self).update_e(sys_vars, replay_memory)
        epi = sys_vars['epi']
        if not (epi % 3) and epi > 15:
            # drop to 1/3 of the current exploration rate
            self.e = max(self.e/3., self.final_e)
        return self.e
