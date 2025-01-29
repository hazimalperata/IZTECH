package src.question;

import java.util.ArrayList;
import java.util.List;

import src.question.type.Audio;
import src.question.type.IType;
import src.question.type.Text;

public class Listening extends Question {
    List<IType> questionTypes;

    public Listening() {
        this(7);
    }

    public Listening(int points) {
        super(points);
        createQuestion();
    }

    private void createQuestion() {
        questionTypes = new ArrayList<IType>();
        IType stringQuestion = new Text();
        IType audioQuestion = new Audio();
        questionTypes.add(stringQuestion);
        questionTypes.add(audioQuestion);
    }


    public List<IType> getQuestionTypes() {
        return questionTypes;
    }

    @Override
    public String toString() {
        return "L";
    }
}
