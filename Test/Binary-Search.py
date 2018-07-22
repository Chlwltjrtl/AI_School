'''
# Binary-Search

Yang-Jae-Innovation-hub 
  Ai- hands-on worker -Test
'''

#import elice_utils


def binary_search(lst, cond):
    '''
    (Optional) 이 함수를 반드시 구현할 필요는 없습니다. 즉, 이 함수는 채점에 포함되지 않습니다.
	이진탐색을 활용하여 lst 내에서 cond 조건을 만족하는 최소의 원소를 반환하는 함수를 작성하세요.
	lst 는 오름차순으로 정렬되어있다고 가정해도 좋습니다.

    ex) binary_search([1,2,3,4,5], lambda x: x>1) returns 2
    '''
    pass


def find_min_square_root(n):
    '''
	q^2 >= n 을 만족하는 최소의 q를 반환하는 함수를 작성하세요.
	경우에 따라 위에 제시되어 있는 binary_search 함수를 작성하여 사용하여도 좋습니다.
    '''
    low, high =1 , n

    while low<=high:
        mid = (low+high)//2

        if mid**2 >= n:
            if (mid-1)**2 < n:
                return mid
            high = mid-1
        else:
            low = mid+1


def main():
    '''
    Do not change this code
    '''
    n = int(input())
    print(find_min_square_root(n))


if __name__ == "__main__":
    main()

