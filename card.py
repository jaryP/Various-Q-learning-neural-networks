import numpy as np
from random import randint


class Card:

    def __init__(self,seme,number):
        self.seme=seme
        self.number=number

    def toBinary(self):
        print(self.number,self.seme)
        final = np.zeros((6))
        c = bin(self.number)[2:]
        print(len(c))
        if len(c) < 5:
            c = c.zfill(5-len(c))
        s = bin(self.seme)[2:]
        if len(s) < 3:
            s = s.zfill(3 - len(s))
        c = np.fromstring(c,'i1') - 48
        s = np.fromstring(s,'i1') - 48
        print(c)
        print(s)
        return np.concatenate((s,c))

    def __str__(self):
        s = ''
        if self.number == 1:
            s+='Asso'
        elif self.number == 2:
            s+='Due'
        elif self.number == 3:
            s += 'Tre'
        elif self.number == 4:
            s += 'Quattro'
        elif self.number == 5:
            s += 'Cinque'
        elif self.number == 6:
            s += 'Sei'
        elif self.number == 7:
            s += 'Sette'
        elif self.number == 8:
            s += 'Otto'
        elif self.number == 9:
            s += 'Nove'
        elif self.number == 10:
            s += 'Dieci'

        s+=' di '

        if self.seme == 0:
            s += 'Denari'
        if self.seme == 1:
            s += 'Coppe'
        if self.seme == 2:
            s += 'Bastoni'
        if self.seme == 3:
            s += 'Spade'
        return s

class Agent:

    def __init__(self, hand, id, type='random'):
        self.type = type
        self.id = id
        self.hand = hand
        self.taken = []
        self.scope = 0
        self.reward = 0

    def move(self,table, verbose=0):

        chosen = randint(0, len(self.hand)-1)
        c = self.hand[chosen]
        self.hand = np.delete(self.hand,chosen)
        possibleSum = table.calculateSums()
        possibleTake = []

        for ps in possibleSum:
            if ps[1] == c.number:
                possibleTake.append(ps)
                #print(c.number,ps[1])

        if len(possibleTake)>0:
            pick = randint(0, len(possibleTake) - 1)
            ps = possibleTake[pick]
            if verbose:
                print('Player ' +str(self.id)+ ' is picking: ',end='')
            self.taken.append(c)

            roundTaken = []
            roundTaken.append(c)

            for ps in ps[0]:
                self.taken.append(ps)
                table.removeCard(ps)
                roundTaken.append(ps)
                if verbose:
                    print(str(ps)+' ('+str(ps.number)+', '+str(ps.seme)+') ' ,end=', ')

            if verbose:
                print('With: '+str(c)+'('+str(c.number)+', '+str(c.seme)+') ')

            if len(table.cards) == 0:
                print('SCOPA DEL GIOCATORE',self.id)
                self.scope +=1

            return roundTaken

        else:
            if verbose:
                print('Player ' +str(self.id)+ ' put '+str(c)+'('+str(c.number)+', '+str(c.seme)+') '+' on the table.')
            table.addCard(c)
            return []

    def isIntelligent(self):
        return self.type == 'intelligent'

    def takeCards(self,cards):
        for c in cards:
            self.taken.append(c)

    def incrementReward(self,r):
        self.reward += r

    def getType(self):
        return self.type

    def getHand(self):
        return self.hand

    def setHand(self,h):
        self.hand = h

    def getId(self):
        return self.id

    def getTaken(self):
        return self.taken

class Game:

    class Table:
        def __init__(self,cards):
            self.cards = cards

        def calculateSums(self):
            import itertools
            l = []
            subs = set()
            for i in range(1,self.cards.size):
                s = set(itertools.combinations(self.cards, i))
                subs.update(s)
            for s in subs:
                sum = 0
                for c in s:
                    sum+=c.number
                if sum <= 10:
                    l.append((s,sum))

            return np.asarray(l)

        def removeCard(self,c):
            i = np.where(self.cards == c)
            self.cards = np.delete(self.cards,i)

        def addCard(self,c):
            self.cards = np.append(self.cards,c)


    def __init__(self):
        self.player = 2
        self.cards = 40
        self.deck = None
        self.table = None
        self.state = np.zeros((4,10))
        pass

    def createDeck(self):
        if self.deck == None:
            l = []
            self.deck = np.empty(40)
            for i in range(1,11):
                for j in range(4):
                    l.append(Card(j,i))
            self.deck = np.asarray(l)
            return self.deck
        else:
            return self.deck

    def calculateScore(self,a1,a2):
        t1 = a1.getTaken()
        t2 = a2.getTaken()
        d = 0
        s = 0
        sb = 0
        p1 = 0
        p2 = 0

        for c in t1:

            if c.seme == 0:
                d+=1

            if c.number == 7:
                s+=1
                if c.seme == 0:
                    sb = 1

        if d>=6:
            p1 +=1
        elif d<=4:
            p2+=1

        if sb == 1:
            p1+=1
        else:
            p2+=1

        if s >= 3:
            p1 += 1
        elif s <= 1:
            p2 += 1

        if len(t1) >20:
            p1+=1
        elif len(t2) >20:
            p2+=1
        print('PRESE:',len(t1))
        p1+=a1.scope
        p2+=a2.scope

        return p1,p2

    def calculatePickReward(self, pick):
        r = 0
        if len(self.table.cards) == 0:
            r +=10
        return len(pick)+r

    def updateState(self,a1,a2,forAgentId=1):

        for c in a1.taken:
            self.state[c.seme][c.number-1] = a1.getId()

        for c in a2.taken:
            self.state[c.seme][c.number-1] = a2.getId()

        for c in self.table.cards:
            self.state[c.seme][c.number-1] = 3

        if a1.getId() == forAgentId:
            for c in a1.getHand():
                self.state[c.seme][c.number - 1] = a1.getId()* -1

        if a2.getId() == forAgentId:
            for c in a2.getHand():
                self.state[c.seme][c.number - 1] = a2.getId() * -1

    def startGame(self, verbose = 1):

        points = [0,0]

        while True:

            d = self.deck
            np.random.shuffle(d)

            last = 0
            agents = np.asarray([Agent(d[:3], 1,type='intelligent'),Agent(d[3:6], 2)])

            a2 = Agent(d[:3], 1,type='intelligent')
            a1 = Agent(d[3:6], 2)

            self.table = self.Table(d[6:10])
            #self.updateState(a1,a2)
            #print(len(self.table.cards))
            d = d[10:]
            #self.updateState(a1,a2)
            if verbose:
                print('Drawing... Remaining cards: ', len(d))

            while True:

                if len(a1.getHand())==0 and len(a2.getHand())==0:
                    a1.setHand(d[:3])
                    a2.setHand(d[3:6])
                    d=d[6:]
                    if verbose:
                        print()
                        print('Drawing... Remaining cards: ', len(d))

                if verbose:
                    print('\nGiocatore 1:\n\t',end='')
                    for c in a1.getHand():
                        print(c,end=', ')

                    print('\nGiocatore 2:\n\t',end='')
                    for c in a2.getHand():
                        print(c, end=', ')

                    print('\nCarte a tavolo:\n\t', end='')
                    for i in self.table.cards:
                        print(i, end=', ')
                    print()

                self.updateState(a1, a2)

                r = a1.move(self.table,verbose)

                if r:
                    last = a1.id
                    if a1.isIntelligent():
                        a1.incrementReward(self.calculatePickReward(r))

                self.updateState(a1, a2)

                r = a2.move(self.table,verbose)
                if r:
                    last = a2.id
                    if a2.isIntelligent():
                        a2.incrementReward(self.calculatePickReward(r))

                self.updateState(a1, a2)

                #exit()
                if False:
                    print(self.state)
                    input("Press Enter to continue...")

                if len(d) == 0 and len(a1.getHand())==0 and len(a2.getHand())==0:

                    print('MANO FINITA')
                    if verbose:
                        print('\nCarte a tavolo:\n\t', end='')

                        for i in self.table.cards:
                            print(i, end=', ')
                        print()

                    if len(self.table.cards) != 0:
                        if last == 1:
                            if verbose:
                                print('Player 1 takes all the remaining cards')
                            if a1.isIntelligent():
                                a1.incrementReward(len(self.table.cards))
                            a1.takeCards(self.table.cards)
                        else:
                            if verbose:
                                print('Player 2 takes all the remaining cards')

                            if a2.isIntelligent():
                                a2.incrementReward(len(self.table.cards))
                            a2.takeCards(self.table.cards)

                    self.table.cards = []
                    self.updateState(a1, a2)
                    #print(self.state)

                    s1, s2 = self.calculateScore(a1,a2)
                    points[0] += s1
                    points[1] += s2
                    print(s1,s2)
                    print(points)
                    input("Press Enter to continue...")

                    #print('Toltal reward: ',a1.reward)

                    break

            if points[0] == points[1] and points[0] == 11:
                print('PAREGGIO')
                break
            elif points[0] >=11:
                print('VINCE GIOCATORE 1')
                break
            elif points[1]>=11:
                print('VINCE GIOCATORE 2')
                break

if __name__ == '__main__':
    g = Game()
    g.createDeck()

    for c in g.deck:
        print(c.number,c.seme,c)

    g.startGame(verbose = 0)
    pass