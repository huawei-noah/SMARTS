import logging
import atexit

from threading import Lock, Thread

from .remote_agent import RemoteAgent, RemoteAgentException


class RemoteAgentBuffer:
    def __init__(self, buffer_size=1):
        self._log = logging.getLogger(self.__class__.__name__)
        self._buffer_size = buffer_size
        self._quiescing = False
        self._agent_buffer = []
        self._agent_buffer_lock = Lock()
        self._replenish_thread = None
        self._start_replenish_thread()
        atexit.register(self.__del__)

    def __del__(self):
        self.destroy()

    def destroy(self):
        self._quiescing = True
        if self._replenish_thread_is_running():
            self._replenish_thread.join()  # wait for the replenisher to finish

        with self._agent_buffer_lock:
            for remote_agent in self._agent_buffer:
                remote_agent.terminate()

    def _replenish_thread_is_running(self):
        return self._replenish_thread is not None and self._replenish_thread.is_alive()

    def _replenish_agents(self):
        # For high-availability, we allow for the possibility that we may create
        # more than `buffer_size` number of agents in the buffer under certain race conditions.
        #
        # To prevent this we would have to put the `RemoteAgent()` creation calls inside the lock
        # context, this would slow down the `acquire_remote_agent` code path.
        fresh_procs = []
        for _ in range(self._buffer_size - len(self._agent_buffer)):
            if self._quiescing:
                return  # early out if we happened to be shutting down midway replenishmen

            try:
                fresh_procs.append(RemoteAgent())
            except RemoteAgentException:
                self._log.error("Failed to initialize remote agent")

        with self._agent_buffer_lock:
            self._agent_buffer.extend(fresh_procs)

    def _start_replenish_thread(self):
        if self._quiescing:
            # not starting thread since we are shutting down
            pass
        elif self._replenish_thread is not None and self._replenish_thread.is_alive():
            # not starting thread since there's already one running
            pass
        else:
            # otherwise start the replenishment thread
            self._replenish_thread = Thread(target=self._replenish_agents, daemon=True)
            self._replenish_thread.start()

    def acquire_remote_agent(self) -> RemoteAgent:
        with self._agent_buffer_lock:
            if len(self._agent_buffer) > 0:
                remote_agent = self._agent_buffer.pop()
            else:
                remote_agent = None

            self._start_replenish_thread()

        if remote_agent is None:
            # Do this here instead of in the else branch above to avoid holding the lock
            # while the RemoteAgent() is being created
            remote_agent = RemoteAgent()

        return remote_agent
