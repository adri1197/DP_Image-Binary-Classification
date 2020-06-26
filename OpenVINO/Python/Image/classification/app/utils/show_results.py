
class ShowResults:

    @staticmethod
    def showOutput(api_type,batch_size,inference_output:list,filename,labels: list):
        for batch,output in enumerate(inference_output):
            for i,inf in enumerate(output):
                inf = inf[0]
                label_id = 1 if inf >= 0.5 else 0
                if api_type == 'sync':
                    print("Image: {} - {:.3f} [{}] ({:.3f}%)".format(filename[(batch_size*batch)+i],inf,labels[label_id],((1-inf)*100)))
                else:
                    print("Batch {} | Image: {} - {:.3f} [{}] ({:.3f}%)".format(batch+1,filename[(batch_size*batch)+i],inf,labels[label_id],((1-inf)*100)))