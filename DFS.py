import numpy as np

class DFS():
    def __init__(self, img):        
        self.img = img
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.objCount = 0
    
    def getObjectCount(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.img[i][j] == 255 and self.visited[i][j] is False:
                    self.objCount += 1
                    self.dfs(i, j)
        
        return self.objCount

    def dfs(self, x, y):
        if x < 0 or x >= self.height or y < 0 or y >= self.width or self.visited[x][y] is True:
            return
        if self.img[x][y] == 255:
            self.visited[x][y] = True
            self.dfs(x-1, y+1)  # 1:30
            self.dfs(x, y+1)    # 3
            self.dfs(x+1, y+1)  # 4:30
            self.dfs(x+1, y)    # 6
            self.dfs(x+1, y-1)  # 7:30
            self.dfs(x, y-1)    # 9
            self.dfs(x-1, y-1)  # 10:30
            self.dfs(x-1, y)    # 12

