package main

import (
	"fmt"
	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/text"
)

func main() {
	// create the channel of data and errors
	stream := make(chan base.TextDatapoint, 40)
	errors := make(chan error)

	model := text.NewNaiveBayes(stream, 3, base.OnlyWordsAndNumbers)

	go model.OnlineLearn(errors)

	stream <- base.TextDatapoint{
		X: "Indian cities look alright",
	}

	stream <- base.TextDatapoint{
		X: "New Delhi, a city in India, gets very hot",
	}

	stream <- base.TextDatapoint{
		X: "Indian food is oftentimes based on vegetables",
	}

	stream <- base.TextDatapoint{
		X: "China is a large country",
	}

	stream <- base.TextDatapoint{
		X: "Chinese food tastes good",
	}

	stream <- base.TextDatapoint{
		X: "Chinese, as a country, has a lot of people in it",
	}

	stream <- base.TextDatapoint{
		X: "Japan makes sushi and cars",
	}

	stream <- base.TextDatapoint{
		X: "Many Japanese people are Buddhist",
	}

	stream <- base.TextDatapoint{
		X: "Japanese architecture looks nice",
	}

	close(stream)

	for {
		err, more := <-errors
		if more {
			fmt.Printf("Error passed: %v", err)
		} else {
			// training is done
			break
		}
	}

	// cast NaiveBayes model to TFIDF
	tf := text.TFIDF(*model)

	greater := tf.TFIDF("sushi", "sushi is my favorite buddhist related food and sushi is fun")
	lesser := tf.TFIDF("buddhist", "sushi is my favorite buddhist related food and sushi is fun")

	fmt.Println(greater, lesser)

	fmt.Println(tf.MostImportantWords("sushi is fun and easy", 3))
}
