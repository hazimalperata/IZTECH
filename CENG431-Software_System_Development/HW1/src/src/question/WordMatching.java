package src.question;

import utils.Rand;

import java.util.HashMap;

import src.question.type.IType;
import src.question.type.Text;



public class WordMatching extends Question {
    HashMap<IType, IType> questionTypes;

    public WordMatching() {
        this(5);
    }

    public WordMatching(int points) {
        super(points);
        createQuestion();
    }

    private void createQuestion() {
        questionTypes = new HashMap<IType, IType>();
        Rand rand = new Rand();
        for (int i = 0; i < rand.getInt(4,8); i++){
            IType key = new Text();
            IType value = new Text();
            questionTypes.put(key, value);
        }
    }


    public HashMap<IType, IType> getQuestionTypes() {
        return questionTypes;
    }


    @Override
    public String toString() {
        return "W";
    }
}
