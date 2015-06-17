#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <stack>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <pthread.h>

#define MAX_STRING_LEN 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define PI 3.14159265358979323846

using namespace std;

typedef float real;

/**hyper parameters for training**/
int thread_num = 1;
int iter = 100;
int vector_dim = 100;
real learn_rate_jt = 0.01;//0.01
real learn_rate_gr = 0.01;
real learn_rate_gr_relation = 0.001;
real gama_jt = 0.5;//for text training
real gama_gr = 1;//for transE

bool learn_jt = false;
bool learn_graph = false;

char run_type[MAX_STRING_LEN];
char jt_input_file[MAX_STRING_LEN];
char gr_input_file[MAX_STRING_LEN];
char model_file_output[MAX_STRING_LEN];
char similarity_test_file[MAX_STRING_LEN];
char topn_output[MAX_STRING_LEN];
char wordvec_input_file[MAX_STRING_LEN];
char model_file_input[MAX_STRING_LEN];
char prt_file[MAX_STRING_LEN];//for initialization

const int max_entry_size = 6000000;     // Maximum 6M entries
const int entry_hash_size = 12000000;   // Maximum 12M, hash size is two times of entry size in case of two many hash crash
const int max_word_size = 1000000;       // Maximum 1M words
const int word_hash_size = 2000000;      // Maximum 2M

std::vector<int> shuffle_vec;
// save the pos of each entry in entry array
int *entry_hash;
// save the pos of each word in word array
int *word_hash;
// vector of entries
real *entry_vec;
// vector of words
real *word_vec;
// vector of relations
real *relation_vec;
// negative sampling pool for joint text training
std::vector<int> word_negative_list;
// save relation map
std::map<std::string,int> rel2id;
std::map<int,std::string> id2rel;

/**statistic infomation**/
int entry_size = 0;
int word_size = 0;
int rel_size = 0;

//struct for count correlation coefficient
struct Id_Score_Rank {
    int id;
    real score;
    int rank;
};

bool compareScore(Id_Score_Rank a, Id_Score_Rank b) {
    return a.score > b.score;
}
bool compareId(Id_Score_Rank a, Id_Score_Rank b) {
    return a.id < b.id;
}
/**store the entry infomation**/
struct entry {
    char *entry_name;
    std::vector<int> word_list;
    int word_num;
};

/** store word information **/
struct word_info {
    char *word;
    int count;
};

struct TestData_Sim {
    std::vector<int> leftindex;
    std::vector<int> rightindex;
    real score;
};

struct Triple {
    int lid;
    int relid;
    int rid;
};

// array to store the infomation of each entry
struct entry *entry_array;
// array to store the infomation of ea
struct word_info *word_array;

//store graph information
std::vector<Triple> triplet;
std::vector<Triple> testtriple;

std::vector<TestData_Sim> test_array_sim;


void split(const std::string& src, const std::string& separator, std::vector<std::string>& dest) {
    std::string str=src;
    std::string substring;
    std::string::size_type start=0,index;

    do
    {
        index=str.find_first_of(separator,start);
        if(index!=std::string::npos)
        {
            substring= str.substr(start,index-start);
            dest.push_back(substring);
            start = str.find_first_not_of(separator,index);
            if (start == std::string::npos) return;
        }
    }while(index != std::string::npos);
    substring=str.substr(start);
    dest.push_back(substring);
}

unsigned int get_entry_hash(const char *entry_name) {
    unsigned long long hash = 1;
    int str_len = strlen(entry_name);
    for (int i = 0; i < str_len; i++) hash = hash * 257 + entry_name[i];
    hash = hash % entry_hash_size;
    return hash;
}

unsigned int get_word_hash(const char *word) {
    unsigned long long hash = 1;
    int str_len = strlen(word);
    for (int i = 0; i < str_len; i++) hash = hash * 257 + word[i];
    hash = hash % word_hash_size;
    return hash;
}

unsigned int get_insert_entry_hash(const char *entry_name) {
    unsigned int hash = get_entry_hash(entry_name);
    while (-1 != entry_hash[hash]) {
        hash = (hash + 1) % entry_hash_size;
    }
    return hash;
}

unsigned int get_insert_word_hash(const char *word) {
    unsigned int hash = get_word_hash(word);
    while (-1 != word_hash[hash]) {
        hash = (hash + 1) % word_hash_size;
    }
    return hash;
}

int search_entry(const char *entry_name) {
    unsigned int hash = get_entry_hash(entry_name);
    while (1) {
        if (entry_hash[hash] == -1) return -1;
        if (!strcmp(entry_name, entry_array[entry_hash[hash]].entry_name)) return hash;
        hash = (hash + 1) % entry_hash_size;
    }
}

int search_word(const char *word) {
    unsigned int hash = get_word_hash(word);
    while (1) {
        if (word_hash[hash] == -1) return -1;
        if (!strcmp(word, word_array[word_hash[hash]].word)) return hash;
        hash = (hash + 1) % word_hash_size;
    }
}

void load_graph(const char *filepath) {
    std::ifstream fin(filepath);
    std::string line;
    int hash, leftid, rightid, relid;
    while(getline(fin, line)) {
        std::vector<std::string> l;
        split(line,"\t",l);
        
        hash = search_entry(l[0].c_str());
        if (-1 == hash) {
            leftid = entry_size;
            hash = get_insert_entry_hash(l[0].c_str());
            entry_hash[hash] = entry_size;
            
            entry_array[entry_size].entry_name = (char *)calloc(l[0].length() + 1, sizeof(char));
            strcpy(entry_array[entry_size].entry_name, l[0].c_str());
            entry_array[entry_size].entry_name[l[0].size()] = '\0';
            entry_size++;
            if (entry_size % 1000 == 0) cout << "entry size: " << entry_size << endl;
        }
        else {
            leftid = entry_hash[hash];
        }
        
        hash = search_entry(l[2].c_str());
        if (-1 == hash) {
            rightid = entry_size;
            hash = get_insert_entry_hash(l[2].c_str());
            entry_hash[hash] = entry_size;
            
            entry_array[entry_size].entry_name = (char *)calloc(l[2].length() + 1, sizeof(char));
            strcpy(entry_array[entry_size].entry_name, l[2].c_str());
            entry_array[entry_size].entry_name[l[2].size()] = '\0';
            entry_size++;
            if (entry_size % 1000 == 0) cout << "entry size: " << entry_size << endl;
        }
        else {
            rightid = entry_hash[hash];
        }
        
        if(rel2id.find(l[1])==rel2id.end())
        {
                relid=rel2id.size();
                rel2id.insert(std::pair<std::string,int>(l[1],relid));
                id2rel.insert(std::pair<int,std::string>(relid,l[1]));
        }
        else
            relid=rel2id[l[1]];
            
        Triple trip = {leftid, relid, rightid};
        triplet.push_back(trip);
    }
    std::random_shuffle(triplet.begin(), triplet.end());
    rel_size = rel2id.size();
}

void add_entry_paragraph(const std::string &entry_info) {
        std::vector<std::string> l;
        split(entry_info,"\t",l);
        std::string entry_name = l[0];
		
        int hash = search_entry(entry_name.c_str());
        int entry_name_hash = -1;
        //cout<<entry_name<<" "<<hash<<endl;
        if (-1 == hash) {
            entry_name_hash = get_insert_entry_hash(entry_name.c_str());
            // save entity name to entry array
            entry_array[entry_size].entry_name = (char *)calloc(entry_name.length() + 1, sizeof(char));
            strcpy(entry_array[entry_size].entry_name, entry_name.c_str());
            entry_array[entry_size].entry_name[entry_name.size()] = '\0';
            entry_hash[entry_name_hash] = entry_size;
            entry_size++;
            if (entry_size % 1000 == 0) cout << "entry size: " << entry_size << endl;
        } else {
            entry_name_hash = hash;
        }
		if(l.size()>1){ 
			std::vector<std::string> paragraph;
			split(l[1]," ",paragraph);
		
			int wordhash = -1;
			//entry_array[entry_hash[entry_name_hash]].word_list = (int *)calloc(paragraph.size(), sizeof(int));
			for (unsigned int i = 0; i < paragraph.size(); i++) {
				std::string word = paragraph[i];
				wordhash = search_word(word.c_str());
				if (-1 != wordhash) {
					word_array[word_hash[wordhash]].count++;
				} else {
					wordhash = get_insert_word_hash(word.c_str());
					word_array[word_size].word = (char *)calloc(word.length() + 1, sizeof(char));
					strcpy(word_array[word_size].word, word.c_str());
					word_array[word_size].word[word.size()] = '\0';
					word_array[word_size].count = 1;
					word_hash[wordhash] = word_size;                
					word_size++; 
				}
				entry_array[entry_hash[entry_name_hash]].word_list.push_back(word_hash[wordhash]);
			}
			entry_array[entry_hash[entry_name_hash]].word_num = paragraph.size();
		} else {
			entry_array[entry_hash[entry_name_hash]].word_num = 0;
		}
}

void load_entry_paragraph(const char *file_path) {
    std::ifstream inStm(file_path);
    std::string line; 
    int num = 0;
    while (getline(inStm, line)) {
        // cout << line << endl;
        add_entry_paragraph(line);
        num++;
        if (num % 100000 == 0) {
            cout << "load_entry_paragraph, line_num: " << num << endl;
        }
    }
    for (int i = 0; i < word_size; ++i) {
        for (int j = 0; j < (int)sqrt((real)word_array[i].count); ++j) word_negative_list.push_back(i);
    }
    cout << "total entry num: " << entry_size << endl;
    inStm.close();
}
        
// init entry vector, entity and category have the same dimension
void init_vector() {
    
    posix_memalign((void **)&entry_vec, 128, (long long)entry_size * vector_dim * sizeof(real));
    if (NULL == entry_vec) std::cerr << "memory allocation fail for entry_vec." << endl;
    posix_memalign((void **)&word_vec, 128, (long long)word_size * vector_dim * sizeof(real));
    if (NULL == word_vec) std::cerr << "memory allocation fail for word_vec." << endl;
    posix_memalign((void **)&relation_vec, 128, (long long)rel_size * vector_dim * sizeof(real));
    if (NULL == relation_vec) std::cerr << "memory allocation fail for relation_vec." << endl;

    // randomize the entry vector
    unsigned long long next_random = 1;
    for (int i = 0; i < entry_size; i++) {
		real sum = 0;
		for (int j = 0; j < vector_dim; j++) {
			next_random = next_random * (unsigned long long)25214903917 + 11;
            entry_vec[i * vector_dim + j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vector_dim;
			sum += entry_vec[i * vector_dim + j] * entry_vec[i * vector_dim + j];
		}
		for (int j = 0; j < vector_dim; j++) entry_vec[i * vector_dim + j] /= sqrt(sum);
    }
    
    for (int i = 0; i < word_size; i++) for (int j = 0; j < vector_dim; j++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        word_vec[i * vector_dim + j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vector_dim;
    }
    
    for (int i = 0; i < rel_size; i++) {
        real sum = 0;
        for (int j = 0; j < vector_dim; j++) {
			relation_vec[i * vector_dim + j] = ((real)rand()/RAND_MAX-0.5)*2*6/sqrt((real)vector_dim);
            sum += relation_vec[i * vector_dim + j] * relation_vec[i * vector_dim + j];
        }
        for (int j = 0; j < vector_dim; j++) relation_vec[i * vector_dim + j] /= sqrt(sum);
    }

}

inline real multiply(real *x, real *y, int dim) {
    real res = 0.0;
    for (int i = 0; i < dim; ++i) res += x[i] * y[i];
    return res;
}

void normalization(real *vector, int dim) {
    real norm = 0;
    for (int i = 0; i < dim; ++i) norm += vector[i] * vector[i];
    norm=sqrt(norm);
    for (int i = 0; i < dim; ++i) vector[i]/=norm;
}


void update_embedding_jt(int entry_id, real &cost, real &word_num) {
    real *input_ent_gradient = (real *)calloc(vector_dim, sizeof(real));
    
    word_num += 30 > entry_array[entry_id].word_num ? entry_array[entry_id].word_num : 30;
	std::random_shuffle(entry_array[entry_id].word_list.begin(),entry_array[entry_id].word_list.end());
	int samplesize = 30 > entry_array[entry_id].word_num ? entry_array[entry_id].word_num : 30;
    for (int i = 0; i < samplesize; ++i) {
        int wid = entry_array[entry_id].word_list[i];
        int corrupt_wid = word_negative_list[rand() % word_negative_list.size()];

        real cost_tmp = gama_jt - multiply(&word_vec[wid * vector_dim], &entry_vec[entry_id * vector_dim], vector_dim) \
                  + multiply(&word_vec[corrupt_wid * vector_dim], &entry_vec[entry_id * vector_dim], vector_dim);

        if (cost_tmp > 0) {
            cost += cost_tmp;
            for (int j = 0; j < vector_dim; j++) {
                input_ent_gradient[j] += -word_vec[wid * vector_dim + j] + word_vec[corrupt_wid * vector_dim + j];
                word_vec[wid * vector_dim + j] -= -learn_rate_jt * entry_vec[entry_id * vector_dim + j];
                word_vec[corrupt_wid * vector_dim + j] -= learn_rate_jt * entry_vec[entry_id * vector_dim + j];
            }
        }
    }

    for (int j = 0; j < vector_dim; ++j) {
        entry_vec[entry_id * vector_dim + j] -= learn_rate_jt * input_ent_gradient[j];
    }
    free(input_ent_gradient);
}

void update_embedding_gr(std::vector<Triple> &thread_triplet, real &cost) {
	int count_train=0;
    for(unsigned int i = 0; i < thread_triplet.size(); ++i) {
        int lid = thread_triplet[i].lid;
        int rid = thread_triplet[i].rid;
        int corrupt_lid, corrupt_rid;
        if(rand()%2 == 0) {
            corrupt_lid = rand()%entry_size;
            corrupt_rid = rid;
        } else {
            corrupt_lid = lid;
            corrupt_rid = rand()%entry_size;
        }
        real *left_embedding = entry_vec + lid * vector_dim;
        real *right_embedding = entry_vec + rid * vector_dim;
        real *corrupt_left_embedding = entry_vec + corrupt_lid * vector_dim;
        real *corrupt_right_embedding = entry_vec + corrupt_rid * vector_dim;
        
        real cost_tmp = gama_gr \
        - multiply(left_embedding, right_embedding, vector_dim) \
        + multiply(corrupt_left_embedding, corrupt_right_embedding, vector_dim);
        if (cost_tmp > 0) {
			count_train++;
            cost += cost_tmp;
            real left_gradient,corrupt_left_gradient,right_gradient,corrupt_right_gradient;
            for (int j = 0; j < vector_dim; ++j) {
                left_gradient = -right_embedding[j];
                right_gradient = -left_embedding[j];
                corrupt_left_gradient =  corrupt_right_embedding[j];
                corrupt_right_gradient = corrupt_left_embedding[j];
                
                left_embedding[j] -= learn_rate_gr * left_gradient;
                right_embedding[j] -= learn_rate_gr * right_gradient;
                corrupt_left_embedding[j] -= learn_rate_gr * corrupt_left_gradient;
                corrupt_right_embedding[j] -= learn_rate_gr * corrupt_right_gradient;
            }   
        }
    }
	cout<<count_train<< " pairs been trained.\n";
}

void update_embedding_relation_transE_sphere(std::vector<Triple> &thread_triplet, real &cost) {
    real *addition = new real [vector_dim];
    real *addition2 = new real [vector_dim];
    for (unsigned int i = 0; i < thread_triplet.size(); ++i) {
        int lid = thread_triplet[i].lid;
        int relid = thread_triplet[i].relid;
        int rid = thread_triplet[i].rid;
        int corrupt_lid, corrupt_rid;
        if(rand()%2 == 0) {
            corrupt_lid = rand()%entry_size;
            corrupt_rid = rid;
        } else {
            corrupt_lid = lid;
            corrupt_rid = rand()%entry_size;
        }
        real *left_embedding = entry_vec + lid * vector_dim;
        real *right_embedding = entry_vec + rid * vector_dim;
        real *rel_embedding = relation_vec + relid * vector_dim;
        real *corrupt_left_embedding = entry_vec + corrupt_lid * vector_dim;
        real *corrupt_right_embedding = entry_vec + corrupt_rid * vector_dim;
        
        for (int j = 0; j < vector_dim; ++j) addition[j] = left_embedding[j] + rel_embedding[j];
        for (int j = 0; j < vector_dim; ++j) addition2[j] = corrupt_left_embedding[j] + rel_embedding[j];
		
        real cost_tmp = gama_gr - multiply(addition, right_embedding, vector_dim) + multiply(addition2, corrupt_right_embedding, vector_dim);
        real left_grad, right_grad, rel_grad, corrupt_left_grad, corrupt_right_grad;
        if (cost_tmp > 0) {
            cost += cost_tmp;
            for(int j = 0; j < vector_dim; ++j) {
                left_grad = -right_embedding[j];
                right_grad = -(left_embedding[j] + rel_embedding[j]);
                corrupt_left_grad = corrupt_right_embedding[j];
                corrupt_right_grad = corrupt_left_embedding[j] + rel_embedding[j];
                rel_grad = -right_embedding[j] + corrupt_right_embedding[j];
                
                left_embedding[j] -= learn_rate_gr * left_grad;
                right_embedding[j] -= learn_rate_gr * right_grad;
                corrupt_left_embedding[j] -= learn_rate_gr * corrupt_left_grad;
                corrupt_right_embedding[j] -= learn_rate_gr * corrupt_right_grad;
                rel_embedding[j] -= learn_rate_gr_relation * rel_grad;
            }
            
        }

    }
    delete [] addition;
    delete [] addition2;
}

void update_embedding_relation_transE(std::vector<Triple> &thread_triplet, real &cost) {
    real *res1 = new real [vector_dim];
    real *res2 = new real [vector_dim];
	real sum1, sum2;
	srand((int)time(0));

    for (unsigned int i = 0; i < thread_triplet.size(); ++i) {
        int lid = thread_triplet[i].lid;
        int relid = thread_triplet[i].relid;
        int rid = thread_triplet[i].rid;
        int corrupt_lid, corrupt_rid;
        if(rand()%2 == 0) {
            corrupt_lid = rand()%entry_size;
            corrupt_rid = rid;
        } else {
            corrupt_lid = lid;
            corrupt_rid = rand()%entry_size;
        }
		
        real *left_embedding = entry_vec + lid * vector_dim;
        real *right_embedding = entry_vec + rid * vector_dim;
        real *rel_embedding = relation_vec + relid * vector_dim;
        real *corrupt_left_embedding = entry_vec + corrupt_lid * vector_dim;
        real *corrupt_right_embedding = entry_vec + corrupt_rid * vector_dim;
        sum1 = 0; 
		sum2 = 0;
        for (int j = 0; j < vector_dim; ++j) {
			res1[j] = left_embedding[j] + rel_embedding[j] - right_embedding[j];
			
			sum1 += fabs(res1[j]);
		}
        for (int j = 0; j < vector_dim; ++j) {
			res2[j] = corrupt_left_embedding[j] + rel_embedding[j] - corrupt_right_embedding[j];
			sum2 += fabs(res2[j]);
		}
		
        real cost_tmp = 2 + sum1 - sum2;

        if (cost_tmp > 0) {
            cost += cost_tmp;
            for(int j = 0; j < vector_dim; ++j) {
				real left_grad, right_grad, corrupt_left_grad, corrupt_right_grad, rel_grad=0;
				if (res1[j] > 0) {
					left_grad = 1;
					rel_grad = 1;
					right_grad = -1;
				} else if (res1[j] < 0) {
					left_grad = -1;
					rel_grad = -1;
					right_grad = 1;
				}
				
				if (res2[j] > 0) {
					corrupt_left_grad = -1;
					rel_grad += -1;
					corrupt_right_grad = 1;
				} else if (res2[j] < 0) {
					corrupt_left_grad = 1;
					rel_grad += 1;
					corrupt_right_grad = -1;
				}
				left_embedding[j] -= learn_rate_gr * left_grad;
				right_embedding[j] -= learn_rate_gr * right_grad;
				corrupt_left_embedding[j] -= learn_rate_gr * corrupt_left_grad;
				corrupt_right_embedding[j] -= learn_rate_gr * corrupt_right_grad;
				rel_embedding[j] -= learn_rate_gr_relation * rel_grad;
            }
        }

    }
    delete [] res1;
    delete [] res2;
}

void output_embedding(const char *filename, const char *run_type) {
    std::fstream fout(filename, std::ios::out);
    fout<<run_type<<" "<<entry_size<<" "<<vector_dim;
    if(rel_size > 0) fout<<" "<<rel_size;
    fout<<endl;
    for (int i = 0; i < entry_size; ++i) {
        fout<<entry_array[i].entry_name;
        for(int j = 0; j < vector_dim; ++j) fout<<" "<<entry_vec[i * vector_dim + j];
        fout<<endl;
    }
    for (int i = 0; i < rel_size; ++i) {
        std::string name = id2rel[i];
        fout<<name;
        for(int j = 0; j < vector_dim; ++j) fout<<" "<<relation_vec[i * vector_dim + j];
        fout<<endl;
    }
    fout.close();
}

void average_word2vec_and_output(){
	bzero(entry_vec, entry_size * vector_dim * sizeof(real));
    for (int i = 0; i < entry_size; ++i) {
        //average
        int countword=0;
		for(int j = 0; j < entry_array[i].word_num; ++j) {			
			int wid = entry_array[i].word_list[j];
			countword+=1;
			for(int k = 0; k < vector_dim; ++k) {
				entry_vec[i * vector_dim + k] += word_vec[wid * vector_dim + k];
			}
		}
		if(countword != 0) {
			for(int k = 0; k < vector_dim; ++k) {
				entry_vec[i * vector_dim + k] /= (real)countword;
			}
		}
    }
}

/*for topN prediction*/
struct heap_node {
    int index;
    real distance;
};

void adjust_heap(heap_node *max_heap, int k, int index) {
    int j = index;
    while (2 * j + 1 < k) {
        int left = 2 * j + 1;
        int right = 2 * j + 2;
        int max_child = -1;
        if (right >= k) {
            max_child = left;
        } else {
            max_child = (max_heap[left].distance > \
                    max_heap[right].distance ? left : right);
        }
        if (max_heap[max_child].distance > max_heap[j].distance) {
            int tmp_index = max_heap[j].index;
            real tmp_distance = max_heap[j].distance;
            max_heap[j].index = max_heap[max_child].index;
            max_heap[j].distance = max_heap[max_child].distance;
            max_heap[max_child].index = tmp_index;
            max_heap[max_child].distance = tmp_distance;
        } else {
            break;
        }
        j = max_child;
    }
}

void build_heap(heap_node *max_heap, real *dis, int k) {
    // initialize the heap
    for (int i = 0; i < k; i++) {
        max_heap[i].index = i;
        max_heap[i].distance = dis[i];
    }

    for (int i = (k - 1) / 2; i >= 0; i--) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int max_child = -1;
        if (right >= k) {
            max_child = left;
        } else {
            max_child = (max_heap[left].distance > \
                    max_heap[right].distance ? left : right);
        }
        if (max_heap[max_child].distance > max_heap[i].distance) {
            int tmp_index = max_heap[i].index;
            real tmp_distance = max_heap[i].distance;
            max_heap[i].index = max_heap[max_child].index;
            max_heap[i].distance = max_heap[max_child].distance;
            max_heap[max_child].index = tmp_index;
            max_heap[max_child].distance = tmp_distance;
            adjust_heap(max_heap, k, max_child);
        }
    }
} 

// return the top k similar entry
// dis: the distance array of the target entry with other entris
// k: top num
std::vector<int> get_similar_entry(real *dis, int n, int k) {
    heap_node *max_heap = (heap_node *)malloc(sizeof(heap_node) * k);
    
    build_heap(max_heap, dis, k);

    // select top k
    for (int i = k; i < n; i++) {
        if (dis[i] >= max_heap[0].distance) continue;
        max_heap[0].index = i;
        max_heap[0].distance = dis[i];
        adjust_heap(max_heap, k, 0);
    }

    std::vector<int> res;
    for (int i = 0; i < k; i++) {
        res.push_back(max_heap[0].index);
        max_heap[0].index = max_heap[k - 1 - i].index;
        max_heap[0].distance = max_heap[k - 1 - i].distance;
        adjust_heap(max_heap, k - 1 - i, 0);
    }
    std::reverse(res.begin(), res.end());
    free(max_heap);
    return res;
}

void topN(int N){
    real *dis_ent=new real [entry_size];
    real *dis_word=new real [word_size];
    std::ofstream fout(topn_output);
    for (int i = 0; i < entry_size; ++i) {
        real *ent_embedding = &entry_vec[i * vector_dim];
        
        for (int j = 0; j < entry_size; ++j)
            dis_ent[j] = -multiply(ent_embedding, &entry_vec[j * vector_dim], vector_dim);
        
        for (int j = 0; j < word_size; ++j)
            dis_word[j] = -multiply(ent_embedding, &word_vec[j * vector_dim], vector_dim);
        
        std::vector<int> res_ent = get_similar_entry(dis_ent, entry_size, N);   
        std::vector<int> res_word = get_similar_entry(dis_word, word_size, N);
        
        fout<<entry_array[i].entry_name<<":"<<endl<<"entry_rank:";
        for (int j = 0; j < N; ++j) fout<<" "<<entry_array[res_ent[j]].entry_name<<",";
        fout<<endl<<"word_rank:";
        for (int j = 0; j < N; ++j) fout<<" "<<word_array[res_word[j]].word<<",";
        fout<<endl;
    }

    for (int i = 0; i < word_size; ++i) {
        real *word_embedding = &word_vec[i * vector_dim];
        
        for (int j = 0; j < entry_size; ++j)
            dis_ent[j] = -multiply(word_embedding, &entry_vec[j * vector_dim], vector_dim);
        for (int j = 0; j < word_size; ++j)
            dis_word[j] = -multiply(word_embedding, &word_vec[j * vector_dim], vector_dim);
        std::vector<int> res_ent = get_similar_entry(dis_ent, entry_size, N);   
        std::vector<int> res_word = get_similar_entry(dis_word, word_size, N);
        fout<<word_array[i].word<<":"<<endl<<"entry_rank:";;
        for (int j = 0; j < N; ++j) fout<<" "<<entry_array[res_ent[j]].entry_name<<",";
        fout<<endl<<"word_rank:";
        for (int j = 0; j < N; ++j) fout<<" "<<word_array[res_word[j]].word<<",";
        fout<<endl;
    }
    
    delete [] dis_ent;
    delete [] dis_word;
    fout.close();
}



struct thread_args_t {
    int entry_num;
    int *entry_p;
    int triplet_num;
    std::vector<Triple> thread_triplet;
    
    real cost_jt;
    real cost_gr;
    real cost_jt_num;
};

// thread_args a ptr pointing to struct thread_args_t
void *train_model_thread(void *thread_args) {
    struct thread_args_t *m_thread_args = (struct thread_args_t *)thread_args;
    int entry_num = m_thread_args->entry_num;
    int *entry_p = m_thread_args->entry_p;
    //int triplet_num = m_thread_args->triplet_num;
    std::vector<Triple> &thread_triplet = m_thread_args->thread_triplet;
    
    if(learn_jt == true) {
        for (int i = 0; i < entry_num; i++) {
			//if(entry_array[entry_p[i]].word_num<100) continue; 
            if (!strcmp(run_type,"jt") ||!strcmp(run_type,"jt_prt") || !strcmp(run_type,"jtgr") || !strcmp(run_type,"jtgr_prt") || !strcmp(run_type,"jtgr_transe_prt"))
				update_embedding_jt(entry_p[i], m_thread_args->cost_jt, m_thread_args->cost_jt_num);
        } 
    }
    if(learn_graph == true) {
        if(!strcmp(run_type,"gr") ||!strcmp(run_type,"gr_prt") || !strcmp(run_type,"jtgr") || !strcmp(run_type,"jtgr_prt"))
            update_embedding_gr(thread_triplet, m_thread_args->cost_gr);
        else if (!strcmp(run_type,"transe_prt") || !strcmp(run_type,"jtgr_transe_prt"))
            //update_embedding_relation_transE(thread_triplet, m_thread_args->cost_gr);
			update_embedding_relation_transE_sphere(thread_triplet, m_thread_args->cost_gr);
    }
    pthread_exit(NULL);
}

void load_similarity_test_file(const char *test_file) {
    std::ifstream in_stm(test_file);
    std::string line;
    while (getline(in_stm, line)) {
        std::vector<std::string> l;
        split(line, "\t", l);
        std::vector<std::string> left_l;
        split(l[0]," ",left_l);
        std::vector<std::string> right_l;
        split(l[1]," ",right_l);
        std::vector<int> leftid;
        for (unsigned int i = 0; i < left_l.size(); ++i) {
            int hash = search_entry(left_l[i].c_str());
            leftid.push_back(entry_hash[hash]);
            if (hash == -1) {
                cout << left_l[i] <<" in the similarity test file does not exist in the model file!" << endl;
                exit(1);
            }
        }
        std::vector<int> rightid;
        for (unsigned int i = 0; i < right_l.size(); ++i) {
            int hash = search_entry(right_l[i].c_str());
            rightid.push_back(entry_hash[hash]);
            if (hash == -1) {
                cout << right_l[i] <<" in the similarity test file does not exist in the model file!" << endl;
                exit(1);
            }
        }
        real score = atof(l[2].c_str());

        TestData_Sim td = {leftid, rightid, score};
        test_array_sim.push_back(td);
    }
    in_stm.close();
}

void test_similarity_euler_per_iter(int iter) {
    cout << "=================iter: " << iter << " similarity test begin.==================" << endl;
	cout << "left_entry right_entry ranking_by_trained_model ranking_by_score_of_answer score_of_answer" << endl;
    // Spearman rank-order correlation coefficient
    std::vector<Id_Score_Rank> score_rank;
    std::vector<Id_Score_Rank> score_rank_answer;
    for (unsigned int i = 0; i < test_array_sim.size(); ++i) {
        real maxcosine=-10000;
        for (unsigned int lefti = 0; lefti < test_array_sim[i].leftindex.size(); ++lefti) {
            for (unsigned int righti = 0; righti < test_array_sim[i].rightindex.size(); ++righti) {
				real score=0;
				for (int j = 0; j < vector_dim; ++j) {
					score -= fabs(entry_vec[test_array_sim[i].leftindex[lefti] * vector_dim + j] \
         					- entry_vec[test_array_sim[i].rightindex[righti] * vector_dim + j]);
				}
			
                if (score > maxcosine) {
                    maxcosine = score;
                }
            }
        }

        Id_Score_Rank isr = {i, maxcosine, 0};
        Id_Score_Rank isr_answer = {i, test_array_sim[i].score, 0};
        score_rank.push_back(isr);
        score_rank_answer.push_back(isr_answer);
    }
    sort(score_rank.begin(), score_rank.end(), compareScore);
    sort(score_rank_answer.begin(), score_rank_answer.end(), compareScore);

    for (unsigned int i = 0; i < score_rank.size(); ++i) {
        score_rank[i].rank = i;
        score_rank_answer[i].rank = i;
    }

    sort(score_rank.begin(),score_rank.end(),compareId);
    sort(score_rank_answer.begin(),score_rank_answer.end(),compareId);

    
    real sum_delta=0;
    for (unsigned int i = 0; i < test_array_sim.size(); ++i) {
        vector<string> entry_name_vec;
		split(entry_array[test_array_sim[i].leftindex[0]].entry_name,"_",entry_name_vec);
		string left_entry_name=entry_name_vec[1];
		entry_name_vec.clear();
		split(entry_array[test_array_sim[i].rightindex[0]].entry_name,"_",entry_name_vec);
		string right_entry_name=entry_name_vec[1];
		entry_name_vec.clear();
        cout << left_entry_name << '\t' \
            << right_entry_name << '\t' \
            << score_rank[i].rank << '\t' << score_rank_answer[i].rank << '\t' \
            << test_array_sim[i].score << endl;
        sum_delta += pow(score_rank[i].rank - score_rank_answer[i].rank, 2);
    }

    real coefficient = 1.0 - 6.0 * sum_delta / (real)test_array_sim.size() / (real)(pow(test_array_sim.size(),2) - 1);
    cout << "Spearman rank-order correlation coefficient: " << coefficient << endl;


    // Pearson correlation coefficient
    real innerproduct_sum=0;
    real predict_sum=0;
    real answer_sum=0;
    real predict_pow_sum=0;
    real answer_pow_sum=0;
    for (unsigned int i=0;i<score_rank.size();++i) {
        predict_sum+=score_rank[i].score;
        predict_pow_sum+=pow(score_rank[i].score, 2);
        answer_sum+=score_rank_answer[i].score;
        answer_pow_sum+=pow(score_rank_answer[i].score, 2);
        innerproduct_sum+=score_rank[i].score * score_rank_answer[i].score;
    }
    coefficient = (score_rank.size() * innerproduct_sum - predict_sum * answer_sum) / \
        sqrt(score_rank.size() * predict_pow_sum - pow(predict_sum, 2)) / \
        sqrt(score_rank.size() * answer_pow_sum - pow(answer_sum, 2));
    cout << "Pearson correlation coefficient: " << coefficient << endl;

    cout << "=================iter: " << iter << " similarity end.==================" << endl << endl;
}

void test_similarity_per_iter(int iter) {
    cout << "=================iter: " << iter << " similarity test begin.==================" << endl;
	cout << "left_entry right_entry ranking_by_trained_model ranking_by_score_of_answer score_of_answer" << endl;
    // Spearman rank-order correlation coefficient
    std::vector<Id_Score_Rank> score_rank;
    std::vector<Id_Score_Rank> score_rank_answer;
    for (unsigned int i = 0; i < test_array_sim.size(); ++i) {
        real maxcosine=-10000;
        for (unsigned int lefti = 0; lefti < test_array_sim[i].leftindex.size(); ++lefti) {
            for (unsigned int righti = 0; righti < test_array_sim[i].rightindex.size(); ++righti) {
                real score = multiply(&entry_vec[test_array_sim[i].leftindex[lefti] * vector_dim], \
                            &entry_vec[test_array_sim[i].rightindex[righti] * vector_dim], vector_dim);
                if (score > maxcosine) {
                    maxcosine = score;
                }
            }
        }

        Id_Score_Rank isr = {i, maxcosine, 0};
        Id_Score_Rank isr_answer = {i, test_array_sim[i].score, 0};
        score_rank.push_back(isr);
        score_rank_answer.push_back(isr_answer);
    }
    sort(score_rank.begin(), score_rank.end(), compareScore);
    sort(score_rank_answer.begin(), score_rank_answer.end(), compareScore);

    for (unsigned int i = 0; i < score_rank.size(); ++i) {
        score_rank[i].rank = i;
        score_rank_answer[i].rank = i;
    }

    sort(score_rank.begin(),score_rank.end(),compareId);
    sort(score_rank_answer.begin(),score_rank_answer.end(),compareId);

    
    real sum_delta=0;
    for (unsigned int i = 0; i < test_array_sim.size(); ++i) {
		vector<string> entry_name_vec;
		split(entry_array[test_array_sim[i].leftindex[0]].entry_name,"_",entry_name_vec);
		string left_entry_name=entry_name_vec[1];
		entry_name_vec.clear();
		split(entry_array[test_array_sim[i].rightindex[0]].entry_name,"_",entry_name_vec);
		string right_entry_name=entry_name_vec[1];
		entry_name_vec.clear();
        cout << left_entry_name << '\t' \
            << right_entry_name << '\t' \
            << score_rank[i].rank << '\t' << score_rank_answer[i].rank << '\t' \
            << test_array_sim[i].score << endl;
        sum_delta += pow(score_rank[i].rank - score_rank_answer[i].rank, 2);
    }

    real coefficient = 1.0 - 6.0 * sum_delta / (real)test_array_sim.size() / (real)(pow(test_array_sim.size(),2) - 1);
    cout << "Spearman rank-order correlation coefficient: " << coefficient << endl;


    // Pearson correlation coefficient
    real innerproduct_sum=0;
    real predict_sum=0;
    real answer_sum=0;
    real predict_pow_sum=0;
    real answer_pow_sum=0;
    for (unsigned int i=0;i<score_rank.size();++i) {
        predict_sum+=score_rank[i].score;
        predict_pow_sum+=pow(score_rank[i].score, 2);
        answer_sum+=score_rank_answer[i].score;
        answer_pow_sum+=pow(score_rank_answer[i].score, 2);
        innerproduct_sum+=score_rank[i].score * score_rank_answer[i].score;
    }
    coefficient = (score_rank.size() * innerproduct_sum - predict_sum * answer_sum) / \
        sqrt(score_rank.size() * predict_pow_sum - pow(predict_sum, 2)) / \
        sqrt(score_rank.size() * answer_pow_sum - pow(answer_sum, 2));
    cout << "Pearson correlation coefficient: " << coefficient << endl;

    cout << "=================iter: " << iter << " similarity end.==================" << endl << endl;
}

void load_word2vec_and_init(const char *filename) { 
	std::ifstream inStm(filename);
    std::string line; 
	getline(inStm, line);
    while (getline(inStm, line)) {
        std::vector<std::string> l;
		split(line, " ", l);
		std::string word = l[0];
		int hash = search_word(word.c_str());
		int wordhash;
		if (-1 == hash) {
            wordhash = get_insert_word_hash(word.c_str());
            // save entity name to entry array
            word_array[word_size].word = (char *)calloc(word.length() + 1, sizeof(char));
            strcpy(word_array[word_size].word, word.c_str());
            word_array[word_size].word[word.size()] = '\0';
            word_hash[wordhash] = word_size;
            word_size++;
            if (word_size % 1000 == 0) cout << "word size: " << word_size << endl;
        }
		
	}
	inStm.close();
	inStm.clear();
	
	init_vector();
	
	inStm.open(filename);
    int num = 0;
	getline(inStm, line);
    while (getline(inStm, line)) {
        std::vector<std::string> l;
		split(line, " ", l);
		std::string word = l[0];
		int hash = search_word(word.c_str());
	
            // save entity name to entry array
		num++;
        int wid = word_hash[hash];
		
		//cout<<word_array[wid].word<<endl;
		real *word_embedding = word_vec + wid * vector_dim;
		for(int i = 0; i < vector_dim; ++i) {
			word_embedding[i] = atof(l[i+1].c_str());
		}
		//cout<<word_vec[word_hash[search_word(word.c_str())] * vector_dim]<<endl;
    }
	cout<<num<<" words founded in word2vec\n";
	
	inStm.close();

}

void load_word2vec_and_init_entry(const char *filename) { 
	std::ifstream inStm(filename);
    std::string line; 
    int num = 0;
	getline(inStm, line);
    while (getline(inStm, line)) {
	//cout<<line<<endl;
        std::vector<std::string> l;
		split(line, " ", l);
		std::string ent = l[0];
		int hash = search_entry(ent.c_str());
		//cout<<ent<< " "<<hash<<endl;
            // save entity name to entry array
		num++;
        int eid = entry_hash[hash];
		real *ent_embedding = entry_vec + eid * vector_dim;
		for(int i = 0; i < vector_dim; ++i) {
			ent_embedding[i] = atof(l[i+1].c_str());
		}
    }
	cout<<num<<" entrys founded in word2vec\n";
	
	inStm.close();

}


void train_model_multithread() {
	
    srand((int)time(0));
    // init shuffle vector
    for (int i = 0; i < entry_size; i++) {
            shuffle_vec.push_back(i);
    }
    std::vector<int> local_shuffle_vec(shuffle_vec);
    std::random_shuffle(local_shuffle_vec.begin(), local_shuffle_vec.end());
    
    load_similarity_test_file(similarity_test_file);

    /*init multi-thread args.*/
    pthread_t *pt = (pthread_t *)malloc(thread_num * sizeof(pthread_t));
    struct thread_args_t *threads_args = (struct thread_args_t *)calloc(thread_num, sizeof(thread_args_t));
    int entry_num_per_thread = local_shuffle_vec.size() / thread_num;
    for (int i = 0; i < thread_num - 1; i++) {
        threads_args[i].entry_num = entry_num_per_thread;
        threads_args[i].entry_p = (int *)malloc(entry_num_per_thread * sizeof(int));
    }
    int offset = entry_num_per_thread * (thread_num - 1);
    int last_entry_num = local_shuffle_vec.size() - offset;
    threads_args[thread_num - 1].entry_num = last_entry_num;
    threads_args[thread_num - 1].entry_p = (int *)malloc(last_entry_num * sizeof(int));

    for (int i = 0; i < thread_num - 1; i++) {
        for (int j = 0; j < entry_num_per_thread; j++) {
            threads_args[i].entry_p[j] = local_shuffle_vec[i * entry_num_per_thread + j];
        }
    }
    for (int i = 0; i < last_entry_num; i++) {
        threads_args[thread_num - 1].entry_p[i] = local_shuffle_vec[i + offset];
    }
    
    
    int triplet_num_per_thread = triplet.size() / thread_num;
    for (int i = 0; i < thread_num - 1; i++) {
        threads_args[i].triplet_num = triplet_num_per_thread;
    }
    offset = triplet_num_per_thread * (thread_num - 1);
    int last_thread_triplet_num = triplet.size() - offset;
    threads_args[thread_num - 1].triplet_num = last_thread_triplet_num;

    for (int i = 0; i < thread_num - 1; i++) {
        for (int j = 0; j < triplet_num_per_thread; j++) {
            threads_args[i].thread_triplet.push_back(triplet[i * triplet_num_per_thread + j]);
        }
    }
    for (int i = 0; i < last_thread_triplet_num; i++) {
        threads_args[thread_num - 1].thread_triplet.push_back(triplet[i + offset]);
    }
    /**********************/
    
    real cost_jt, cost_gr, countword;
    int work_iter = 0;
    while (work_iter < iter) {

        
        
        cost_jt = 0;
        cost_gr = 0;
        countword = 0;
        //training link multi-thread
        for (int i = 0; i < thread_num; i++) pthread_create(&pt[i], NULL, \
            train_model_thread, (void *)(threads_args + i));
        for (int i = 0; i < thread_num; i++) pthread_join(pt[i], NULL);

        for (int i = 0; i < thread_num; i++) {
            cost_jt += threads_args[i].cost_jt;
            cost_gr += threads_args[i].cost_gr;
            countword += threads_args[i].cost_jt_num;
        }
		
		//fix ill-defined object function by limit the value of each dimension. 
		for(int i = 0; i < entry_size; ++i) {
			for(int j = 0; j < vector_dim; ++j) {
				if(entry_vec[i * vector_dim + j] > 5) {
				 entry_vec[i * vector_dim + j]=5;
				 }
				if(entry_vec[i * vector_dim + j] < -5) {
				entry_vec[i * vector_dim + j]=-5;
				}
			}
		}
        
        cout << "iter: " << work_iter << "\tcost_w: " << cost_jt / countword  << "\tcost_r:" <<cost_gr / triplet.size() << endl;
        
        for (int i = 0; i < thread_num; i++) {
            threads_args[i].cost_jt = 0;
            threads_args[i].cost_gr = 0;
            threads_args[i].cost_jt_num = 0;
        }
		
		test_similarity_per_iter(work_iter);
        work_iter++;

    }

    output_embedding(model_file_output,run_type);
    topN(50);//This can be changed
}

void load_similarity_modelfile(const char *model_file) {
	std::string line;
	std::vector<std::string> l;
	long long hash;
	
	std::ifstream input(model_file);
	getline(input, line);
	while(getline(input,line)) {
        l.clear();
        split(line, " ", l);
        hash = search_entry(l[0].c_str());
        int entry_name_hash = -1;
        //cout<<entry_name<<" "<<hash<<endl;
        if (-1 == hash) {
            entry_name_hash = get_insert_entry_hash(l[0].c_str());
            // save entity name to entry array
            entry_array[entry_size].entry_name = (char *)calloc(l[0].length() + 1, sizeof(char));
            strcpy(entry_array[entry_size].entry_name, l[0].c_str());
            entry_array[entry_size].entry_name[l[0].size()] = '\0';
            entry_hash[entry_name_hash] = entry_size;
            entry_size++;
            if (entry_size % 1000 == 0) cout << "entry size: " << entry_size << endl;
        }
    } 
	input.close();
    input.clear();
	
    input.open(model_file);
    getline(input, line);
	l.clear();
    split(line, " ", l);
    vector_dim = atoi(l[2].c_str());
	//cout<<atoi(l[2].c_str())<<endl;
    entry_vec = new real [entry_size * vector_dim];
    for(int i = 0; i < entry_size; ++i) {
        getline(input, line);
        l.clear();
        split(line, " ", l);
        hash = search_entry(l[0].c_str());
        for (int j = 0; j < vector_dim; ++j) {
			entry_vec[entry_hash[hash] * vector_dim + j] = atof(l[j + 1].c_str());
			//cout<<entry_vec[entry_hash[hash] * vector_dim + j]<<" "<<atof(l[j + 1].c_str())<<" ";
		}
		//cout<<endl;
    } 
    
    input.close();
    input.clear();
}

void test_model_similarity() {
	cout << "left_entry right_entry ranking_by_trained_model ranking_by_score_of_answer score_of_answer" << endl;
    // Spearman rank-order correlation coefficient
    std::vector<Id_Score_Rank> score_rank;
    std::vector<Id_Score_Rank> score_rank_answer;
    for (unsigned int i = 0; i < test_array_sim.size(); ++i) {
        real maxcosine=-10;
        for (unsigned int lefti = 0; lefti < test_array_sim[i].leftindex.size(); ++lefti) {
            for (unsigned int righti = 0; righti < test_array_sim[i].rightindex.size(); ++righti) {
                real score = multiply(&entry_vec[test_array_sim[i].leftindex[lefti] * vector_dim], \
                            &entry_vec[test_array_sim[i].rightindex[righti] * vector_dim], vector_dim);
                if (score > maxcosine) {
                    maxcosine = score;
                }
            }
        }

        Id_Score_Rank isr = {i, maxcosine, 0};
        Id_Score_Rank isr_answer = {i, test_array_sim[i].score, 0};
        score_rank.push_back(isr);
        score_rank_answer.push_back(isr_answer);
    }
    sort(score_rank.begin(), score_rank.end(), compareScore);
    sort(score_rank_answer.begin(), score_rank_answer.end(), compareScore);

    for (unsigned int i = 0; i < score_rank.size(); ++i) {
        score_rank[i].rank = i;
        score_rank_answer[i].rank = i;
    }
	
    sort(score_rank.begin(),score_rank.end(),compareId);
    sort(score_rank_answer.begin(),score_rank_answer.end(),compareId);

    
    real sum_delta=0;
    for (unsigned int i = 0; i < test_array_sim.size(); ++i) {
        vector<string> entry_name_vec;
		split(entry_array[test_array_sim[i].leftindex[0]].entry_name,"_",entry_name_vec);
		string left_entry_name=entry_name_vec[1];
		entry_name_vec.clear();
		split(entry_array[test_array_sim[i].rightindex[0]].entry_name,"_",entry_name_vec);
		string right_entry_name=entry_name_vec[1];
		entry_name_vec.clear();
        cout << left_entry_name << '\t' \
            << right_entry_name << '\t' \
            << score_rank[i].rank << '\t' << score_rank_answer[i].rank << '\t' \
            << test_array_sim[i].score << endl;
        sum_delta += pow(score_rank[i].rank - score_rank_answer[i].rank, 2);
    }

    real coefficient = 1.0 - 6.0 * sum_delta / (real)test_array_sim.size() / (real)(pow(test_array_sim.size(),2) - 1);
    cout << "Spearman rank-order correlation coefficient: " << coefficient << endl;
    
    //Pearson correlation coefficient
    real innerproduct_sum=0;
    real predict_sum=0;
    real answer_sum=0;
    real predict_pow_sum=0;
    real answer_pow_sum=0;
    for (unsigned int i=0;i<score_rank.size();++i) {
        predict_sum+=score_rank[i].score;
        predict_pow_sum+=pow(score_rank[i].score, 2);
        answer_sum+=score_rank_answer[i].score;
        answer_pow_sum+=pow(score_rank_answer[i].score, 2);
        innerproduct_sum+=score_rank[i].score * score_rank_answer[i].score;
    }
    coefficient = (score_rank.size() * innerproduct_sum - predict_sum * answer_sum) / \
                    sqrt(score_rank.size() * predict_pow_sum - pow(predict_sum, 2)) / \
                    sqrt(score_rank.size() * answer_pow_sum - pow(answer_sum, 2));
    
    cout<<"The Pearson correlation coefficient between prediction and answer is around "<<coefficient<<endl;
    delete [] entry_vec;
}

void test_model_similarity_euler() {
    // Spearman rank-order correlation coefficient
    std::vector<Id_Score_Rank> score_rank;
    std::vector<Id_Score_Rank> score_rank_answer;
    for (unsigned int i = 0; i < test_array_sim.size(); ++i) {
        real maxcosine=-10;
        for (unsigned int lefti = 0; lefti < test_array_sim[i].leftindex.size(); ++lefti) {
            for (unsigned int righti = 0; righti < test_array_sim[i].rightindex.size(); ++righti) {
                real score = 0;
				for(int k = 0; k < vector_dim; ++k) 
				score += -fabs(entry_vec[test_array_sim[i].leftindex[lefti] * vector_dim + k] - entry_vec[test_array_sim[i].rightindex[righti] * vector_dim + k]);
                if (score > maxcosine) {
                    maxcosine = score;
                }
            }
        }

        Id_Score_Rank isr = {i, maxcosine, 0};
        Id_Score_Rank isr_answer = {i, test_array_sim[i].score, 0};
        score_rank.push_back(isr);
        score_rank_answer.push_back(isr_answer);
    }
    sort(score_rank.begin(), score_rank.end(), compareScore);
    sort(score_rank_answer.begin(), score_rank_answer.end(), compareScore);

    for (unsigned int i = 0; i < score_rank.size(); ++i) {
        score_rank[i].rank = i;
        score_rank_answer[i].rank = i;
    }

    sort(score_rank.begin(),score_rank.end(),compareId);
    sort(score_rank_answer.begin(),score_rank_answer.end(),compareId);

    
    real sum_delta=0;
    for (unsigned int i = 0; i < test_array_sim.size(); ++i) {
        cout << entry_array[test_array_sim[i].leftindex[0]].entry_name << '\t' \
            << entry_array[test_array_sim[i].rightindex[0]].entry_name << '\t' \
            << score_rank[i].rank << '\t' << score_rank_answer[i].rank << '\t' \
            << test_array_sim[i].score << endl;
        sum_delta += pow(score_rank[i].rank - score_rank_answer[i].rank, 2);
    }

    real coefficient = 1.0 - 6.0 * sum_delta / (real)test_array_sim.size() / (real)(pow(test_array_sim.size(),2) - 1);
    cout << "Spearman rank-order correlation coefficient: " << coefficient << endl;
    
    //Pearson correlation coefficient
    real innerproduct_sum=0;
    real predict_sum=0;
    real answer_sum=0;
    real predict_pow_sum=0;
    real answer_pow_sum=0;
    for (unsigned int i=0;i<score_rank.size();++i) {
        predict_sum+=score_rank[i].score;
        predict_pow_sum+=pow(score_rank[i].score, 2);
        answer_sum+=score_rank_answer[i].score;
        answer_pow_sum+=pow(score_rank_answer[i].score, 2);
        innerproduct_sum+=score_rank[i].score * score_rank_answer[i].score;
    }
    coefficient = (score_rank.size() * innerproduct_sum - predict_sum * answer_sum) / \
                    sqrt(score_rank.size() * predict_pow_sum - pow(predict_sum, 2)) / \
                    sqrt(score_rank.size() * answer_pow_sum - pow(answer_sum, 2));
    
    cout<<"The Pearson correlation coefficient between prediction and answer is around "<<coefficient<<endl;
    delete [] entry_vec;
}

int arg_pos(const char *arg, int argc, char **argv) {
    int i = 1;
    for (; i < argc; i++) {
        if (!strcmp(arg, argv[i])) {
            if (i == argc - 1) {
                std::cerr << "argument missing for: " << arg << endl;
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char *argv[]) {
    int i;
    if (argc == 1) {
        cout << "JTGR tool kit v1.0\n\n";
        cout << "Options:\n";
        cout << "Parameters for training:\n";
        cout << "\t-type <Option>\n";
        cout << "\t\trunning mode. <Option> includes:\n"
		     << "\t\t'test-similarity','jt','gr','jt_prt','gr_prt','transe_prt','jtgr','jtgr_prt','jtgr_transe_prt','lt_mean'\n";
		cout << "\t-jt_input <file>\n";
        cout << "\t\tUse text data <file> to train the joint text model\n";
        cout << "\t-gr_input <file>\n";
        cout << "\t\tUse triplet data <file> to train the relation graph model\n";
        cout << "\t-model_output <directory>\n";
        cout << "\t\tName of output model file, when the training finished, model of the final iteration will be output;\n";
		cout << "\t-word2vec_input <file>\n";
		cout << "\t\tInput file of word2vec to initialize word vector for lt_mean;\n";
		cout << "\t-model_input <file>\n";
		cout << "\t\tInput file for test similarity;\n";
		cout << "\t-prt <file>\n";
		cout << "\t\tInput file of word2vec to initialize jt or gr;\n";
        cout << "\t-sim_test_input <file>\n";
        cout << "\t\tInput file of test dataset for similarity\n";
        cout << "\t-topn_output <file>\n";
        cout << "\t\tWhen training finished ,program will output a file containing top20 similar list for each entry\n";
        cout << "\t-threads <int>\n";
        cout << "\t\tUse <int> threads (default 1)\n";
        cout << "\t-iter <int>\n";
        cout << "\t\tRun more training iterations (default 200)\n";
        cout << "\t-dim <int>\n";
        cout << "\t\tDimension for the vectors (default 100)\n";
        cout << "\t-learn_rate_graph <float>\n";
        cout << "\t\tLearning rate for graph learning (default 0.01)\n";
        cout << "\t-learn_rate_graph_relation <float>\n";
        cout << "\t\tLearning rate for relation training on relation vector using TransE (default 0.001)\n";
        cout << "\t-learn_rate_text <float>\n";
        cout << "\t\tLearning rate for text learning(default 0.01)\n";
        cout << "\nExample:\n";
        cout << "./JTGR -type jt -jt_input data/train.wordnet-noun.wikipage.filter -sim_test_input data/test.sim-301 "
			 << "-threads 7 -iter 200 -dim 100 -learn_rate_text 0.01\n";
        cout << "./JTGR -type gr -gr_input data/train.wordnet-noun.pairs -sim_test_input data/test.sim-301 "
             <<"-threads 7 -iter 200 -dim 100 -learn_rate_graph 0.01\n";
        cout << "./JTGR -type jt_prt -jt_input data/train.wordnet-noun.wikipage.filter -prt data/word2vec.wordnet-noun.prt.100 -sim_test_input data/test.sim-301 "
		     << "-threads 7 -iter 200 -dim 100 -learn_rate_text 0.01\n";
		cout << "./JTGR -type gr_prt -gr_input data/train.wordnet-noun.pairs -prt data/word2vec.wordnet-noun.prt.100 -sim_test_input data/test.sim-301 "
		     << "-threads 7 -iter 200 -dim 100 -learn_rate_graph 0.01\n";
        cout << "./JTGR -type transe_prt -gr_input data/train.wordnet-noun.pairs -prt data/word2vec.wordnet-noun.prt.100 -sim_test_input data/test.sim-301 "
		     << "-threads 7 -iter 200 -dim 100 -learn_rate_graph 0.01 -learn_rate_graph_relation 0.001\n";
		cout << "./JTGR -type jtgr -gr_input data/train.yago-animal.pairs -jt_input data/train.yago-animal.wikipage.filter -sim_test_input data/test.animal-143 "
		     << "-threads 7 -iter 200 -dim 100 -learn_rate_graph 0.01 -learn_rate_text 0.01\n";
		cout << "./JTGR -type jtgr_prt -gr_input data/train.yago-animal.pairs -jt_input data/train.yago-animal.wikipage.filter -sim_test_input data/test.animal-143 "
		     << "-prt data/word2vec.yago-animal.prt.100 -threads 7 -iter 200 -dim 100 -learn_rate_graph 0.01 -learn_rate_text 0.01\n";
		cout << "./JTGR -type jtgr_transe_prt -gr_input data/train.yago-animal.pairs -jt_input data/train.yago-animal.wikipage.filter -sim_test_input data/test.animal-143 "
		     << "-prt data/word2vec.yago-animal.prt.100 -threads 7 -iter 200 -dim 100 -learn_rate_graph 0.01 -learn_rate_text 0.01 -learn_rate_graph_relation 0.001\n";
		cout << "./JTGR -type lt_mean -jt_input data/train.yago-animal.wikipage.filter -word2vec_input word2vec.100 -sim_test_input data/test.animal-143 -dim 100\n";
        cout << "./JTGR -type test-similarity -model_input model_demo/JTGR_vector -sim_test_input data/test.sim-301 \n";
		cout << "JTGR -type test-similarity -model_input data/word2vec.wordnet-noun.prt.100 -sim_test_input data/test.sim-301 \n";
		return 0;
    }

    if ((i = arg_pos("-type", argc, argv)) > 0) {
        strcpy(run_type,argv[i + 1]);
    }
	if ((i = arg_pos("-word2vec_input", argc, argv)) > 0) {
        strcpy(wordvec_input_file, argv[i + 1]);
    }
    if ((i = arg_pos("-jt_input", argc, argv)) > 0) {
        strcpy(jt_input_file, argv[i + 1]);
    }
    if ((i = arg_pos("-gr_input", argc, argv)) > 0) {
        strcpy(gr_input_file, argv[i + 1]);
    }
    if ((i = arg_pos("-model_output",argc, argv)) > 0) {
        strcpy(model_file_output,argv[i + 1]);
    }
	if ((i = arg_pos("-prt",argc, argv)) > 0) {
        strcpy(prt_file,argv[i + 1]);
    }
	if ((i = arg_pos("-model_input",argc, argv)) > 0) {
        strcpy(model_file_input,argv[i + 1]);
    }
    if ((i = arg_pos("-sim_test_input",argc, argv)) > 0) {
        strcpy(similarity_test_file,argv[i + 1]);
    }
    if ((i = arg_pos("-threads", argc, argv)) > 0) {
        thread_num = atoi(argv[i + 1]);
    }
    if ((i = arg_pos("-iter", argc, argv)) > 0) {
        iter = atoi(argv[i + 1]);
    }
    if ((i = arg_pos("-dim",argc, argv)) > 0) {
        vector_dim = atoi(argv[i + 1]);
    }
    if ((i = arg_pos("-learn_rate_text", argc, argv)) > 0) {
        learn_rate_jt = atof(argv[i + 1]);
    }
    if ((i = arg_pos("-learn_rate_graph", argc, argv)) > 0) {
        learn_rate_gr = atof(argv[i + 1]);
    }
    if ((i = arg_pos("-learn_rate_graph_relation", argc, argv)) > 0) {
        learn_rate_gr_relation = atof(argv[i + 1]);
    }
    if ((i = arg_pos("-topn_output", argc, argv)) > 0) {
        strcpy(topn_output, argv[i + 1]);
    }
    
    entry_array = (struct entry *)calloc(max_entry_size, sizeof(struct entry));
    entry_hash = (int *)malloc(entry_hash_size * sizeof(int));
    for (int i = 0; i < entry_hash_size; i++) entry_hash[i] = -1;
    
    word_array = (struct word_info *)calloc(max_word_size, sizeof(struct word_info));
    word_hash = (int *)malloc(entry_hash_size * sizeof(int));
    for (int i = 0; i < word_hash_size; i++) word_hash[i] = -1;

    if (!strcmp(run_type,"test-similarity")) {
            load_similarity_modelfile(model_file_input);
            load_similarity_test_file(similarity_test_file);
            test_model_similarity();
    } else if (!strcmp(run_type,"jt")) { //after this, using fast_tune
			load_entry_paragraph(jt_input_file);
			learn_jt = true;
			init_vector();
            train_model_multithread();
    } else if (!strcmp(run_type,"jt_prt") ) {
			load_entry_paragraph(jt_input_file);
			learn_jt = true;
			init_vector();
			load_word2vec_and_init_entry(prt_file);
            train_model_multithread();
    } else if (!strcmp(run_type,"gr")) {
            load_graph(gr_input_file);
            learn_graph = true;
			init_vector();
            train_model_multithread();
    } else if (!strcmp(run_type,"gr_prt")) {
            load_graph(gr_input_file);
            learn_graph = true;
			init_vector();
			load_word2vec_and_init_entry(prt_file); 
            train_model_multithread();
    } else if (!strcmp(run_type,"transe_prt")) {
            load_graph(gr_input_file);
            learn_graph = true;
			init_vector();
			load_word2vec_and_init_entry(prt_file); 
            train_model_multithread();
    } else if (!strcmp(run_type,"jtgr")) {
			load_entry_paragraph(jt_input_file);
            load_graph(gr_input_file);
            learn_jt = true;
            learn_graph = true;
			init_vector();
            train_model_multithread();
    } else if (!strcmp(run_type,"jtgr_prt")) {
			load_entry_paragraph(jt_input_file);
            load_graph(gr_input_file);
            learn_jt = true;
            learn_graph = true;
			init_vector();
			load_word2vec_and_init_entry(prt_file);
            train_model_multithread();
    } else if (!strcmp(run_type,"jtgr_transe_prt")) {
			load_entry_paragraph(jt_input_file);
            load_graph(gr_input_file);
            learn_jt = true;
            learn_graph = true;
			init_vector();
			load_word2vec_and_init_entry(prt_file);
            train_model_multithread();
    } else if (!strcmp(run_type,"lt_mean") ) {
            load_entry_paragraph(jt_input_file);
			load_word2vec_and_init(wordvec_input_file);
			average_word2vec_and_output();
			load_similarity_test_file(similarity_test_file);
			test_similarity_per_iter(0);
			
    }  else {
            cout << "error model type." << endl;
            exit(0);      
    }
    return 0;
}
