#!/usr/bin/env ruby
if ARGV.size < 3
  puts "Usage: #{__FILE__} data_file_1 data_file_2 iter_num"
end

filename_1 = ARGV[0]
filename_2 = ARGV[1]
iter_num = ARGV[2]

file_1_data = File.open(filename_1).read
file_2_data = File.open(filename_2).read

file_1_speed = file_1_data.lines.map{|l| l.split(",")}.select{|array| array[1]==iter_num}
file_2_speed = file_2_data.lines.map{|l| l.split(",")}.select{|array| array[1]==iter_num}

puts file_1_speed
puts '---'
puts file_2_speed
puts '---'
puts "layers,speedup"
file_1_speed.size.times do |i|
  #i = file_1_speed.size - 1 - i
  print file_1_speed[i][0]
  print ","
  puts (file_1_speed[i][2].to_f / file_2_speed[i][2].to_f * 100).round - 100 
end
