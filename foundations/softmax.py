import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        ma=np.max(z)

        ans=np.zeros(len(z))

        den=0
        for j in range(len(z)):
            den+=(np.e**(z[j]-ma))

        for i in range(len(z)):
            num=np.e**(z[i]-ma)
            ans[i]=num/den

        return np.round(ans,4)