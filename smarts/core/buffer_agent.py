# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import abc
from abc import abstractmethod
from smarts.zoo.agent_spec import AgentSpec

class BufferAgent(metaclass = abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls,  subclass):
        return  (hasattr(subclass, 'act') and 
                callable(subclass.act) and
                hasattr(subclass, 'start') and 
                callable(subclass.start) and
                hasattr(subclass, 'terminate') and 
                callable(subclass.terminate))
    
    @abstractmethod
    def act(self, obs):
        raise NotImplementedError

    @abstractmethod
    def start(self, agent_spec: AgentSpec):
        raise NotImplementedError

    @abstractmethod
    def terminate(self):
        raise NotImplementedError