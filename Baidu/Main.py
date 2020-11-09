from API.token import Token
from API.face import Face


def main():
    token = Token()
    # 人脸检测
    result = Face(token.getToken()).detection("8.jpg")
    sex = {"female": "女性", "male": "男性"}
    print("图片：%s.jpg，年龄：%s，颜值评分：%s，性别：%s" % (8, result["age"], result["beauty"], sex[result["gender"]["type"]]))


if __name__ == "__main__":
    main()

# while (True):
#     result, pos_prob, neg_prob = token.sentiment(input())
#     print("情感分析结果：%s，正面可信度：%s，负面可信度：%s" % (result, pos_prob, neg_prob))
